import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.distributions import Dirichlet

from src.models.components.cvae.flow_utils import build_rqs_flow, set_actnorm_identity
from src.models.components.cvae.simplex_utils import (
    LearnedOrthogonal,
    alr,
    dirichlet_dequantize_with_pad,
    helmert_basis,
    ilr,
    inv_ilr,
    inv_alr,
    logdet_jac_ilr,
    logdet_jac_alr,
)
from src.models.components.cvae.transformer_components import CausalTransformerEncoder, make_causal_mask


class TransformerFlowDecoderSimplex(nn.Module):
    """
    Transformer + conditional flow over [ALR(a) || ALR(r) || log t].
    Training: pure teacher forcing (no scheduled sampling).
    Inference: autoregressive generation.
    """
    def __init__(
        self,
        trace_attributes,
        num_activities,
        num_resources,
        max_trace_length,
        num_transformer_layers,
        cf_dim,
        t_dim,
        z_dim,
        c_dim,
        dropout_p,
        tot_attr_e_dim,
        is_conditional,
        device,
        # Flow config
        flow_layers=10,
        use_alternating_mask=True,
        flow_num_bins=24,
        flow_tail_bound=8.0,
        flow_cond_hidden=256,
        flow_cond_blocks=2,
        flow_cond_dropout=0.1,
        flow_use_random_perm_each_layer=True,
        # Transformer config
        num_heads=8,
        use_film = False,
        use_alibi = True,
        # Modeling choices
        use_logspace_time=True,
        use_pre_rotation=True,
        rotation_reflections=2,
        use_pos_ratio=True,
        use_ilr = True,
        use_weighted_schedule=False,
        # Dequantization
        dirichlet_alpha_main=60.0,
        dirichlet_alpha_noise=0.5,
        # Data
        pad_activity_idx=17,
        pad_resource_idx=26,
    ):
        super().__init__()
        self.trace_attributes = trace_attributes
        self.A = num_activities
        self.R = num_resources
        self.Tmax = max_trace_length
        self.L = num_transformer_layers
        self.is_conditional = is_conditional
        self.has_trace_attributes = tot_attr_e_dim > 0
        self.t_dim = t_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.device = device
        
        # ----- Flow hyperparams -----
        self.flow_layers = flow_layers
        self.use_alternating_mask = use_alternating_mask
        self.flow_num_bins = flow_num_bins
        self.flow_tail_bound = flow_tail_bound
        self.flow_cond_hidden = flow_cond_hidden
        self.flow_cond_blocks = flow_cond_blocks
        self.flow_cond_dropout = flow_cond_dropout
        self.flow_use_random_perm_each_layer = flow_use_random_perm_each_layer

        # ----- Transformer hyperparams -----
        self.num_heads = num_heads
        self.cf_dim = cf_dim  # transformer embedding dim
        self.use_film = use_film
        self.use_alibi = use_alibi
        
        # ----- Modeling choices -----
        
        self.use_logspace_time = use_logspace_time
        self.use_pre_rotation = use_pre_rotation
        self.rotation_reflections = rotation_reflections
        self.use_pos_ratio = use_pos_ratio
        self.use_weighted_schedule = use_weighted_schedule
        self.use_ilr = use_ilr

        # ----- Dequantization -----
        self.alpha_main = dirichlet_alpha_main
        self.alpha_noise = dirichlet_alpha_noise
        
        
        self.pad_activity_idx = pad_activity_idx
        self.pad_resource_idx = pad_resource_idx  # set externally if needed
        

        self.ua_dim = self.A - 1
        self.ur_dim = self.R - 1
        
        if self.use_pre_rotation:
            self.pre_rot_a = LearnedOrthogonal(self.ua_dim, k=self.rotation_reflections)
            self.pre_rot_r = LearnedOrthogonal(self.ur_dim, k=self.rotation_reflections)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # Latent -> trace embedding
        concat_z = z_dim + c_dim if is_conditional else z_dim
        self.z2t = nn.Linear(concat_z, t_dim)

        # Trace-level attribute heads
        self.t2attr = nn.ModuleDict()
        for trace_attr in self.trace_attributes:
            hd = t_dim // 2
            if trace_attr["type"] == "categorical":
                self.t2attr[trace_attr["name"]] = nn.Sequential(
                    nn.Linear(t_dim, hd), nn.ReLU(), self.dropout,
                    nn.Linear(hd, len(trace_attr["possible_values"]))
                )
            elif trace_attr["type"] == "numerical":
                self.t2attr[trace_attr["name"]] = nn.Sequential(
                    nn.Linear(t_dim, hd), nn.ReLU(), self.dropout,
                    nn.Linear(hd, 1)
                )
            else:
                raise ValueError("Unknown trace attribute type")

        # Positional encodings
        self.register_buffer(
            "positional_encoding",
            self._create_positional_encoding(self.Tmax + 1, cf_dim)
        )

        # Token projection: [t_rec, ua, ur, log_dt] -> cf_dim
        if self.use_pos_ratio:
            self.input_projection = nn.Linear(t_dim + self.ua_dim + self.ur_dim + 1 + 1, cf_dim)
        else:
            self.input_projection = nn.Linear(t_dim + self.ua_dim + self.ur_dim + 1, cf_dim)


        # Causal mask
        self.register_buffer("causal_mask", self._generate_square_subsequent_mask(self.Tmax + 1))
        self.register_buffer("causal_mask_add", make_causal_mask(self.Tmax + 1, device), persistent=False)
        
        self.register_buffer("ilr_basis_act", helmert_basis(self.A, self.device))
        self.register_buffer("ilr_basis_res", helmert_basis(self.R, self.device))

        use_film: bool = getattr(self, "use_film", False)  # or pass as __init__ arg
        film_ctx_dim =  t_dim  # same context you feed to the flow (see below)

        self.transformer_encoder = CausalTransformerEncoder(
            num_layers=self.L,
            d_model=cf_dim,
            nhead=num_heads,
            dim_ff= 4*cf_dim,
            dropout=dropout_p,
            use_film=use_film,
            ctx_dim=(film_ctx_dim if use_film else 0),
            use_alibi=self.use_alibi,  # CHANGE
)

        # Conditional flow over [ua || ur || log t]
        tuple_dim = self.ua_dim + self.ur_dim +1        

        # Context dim for flow: transformer context (+ t_rec if enabled)
        self.context_dim = cf_dim + t_dim  # always include t_rec; simple & stable

        
        
        self.flow = build_rqs_flow(
                in_features=tuple_dim,
                context_features=self.context_dim,
                layers=self.flow_layers,     # default 12 if you don't have self.flow_layers
                num_bins=self.flow_num_bins, # default 24
                tail_bound=self.flow_tail_bound, # 8 is good 
                cond_hidden=self.flow_cond_hidden,
                cond_blocks=self.flow_cond_blocks,
                cond_dropout=self.flow_cond_dropout,
                use_alternating_mask=self.use_alternating_mask,
                use_random_perm_each_layer=self.flow_use_random_perm_each_layer,
        )
        

        
        
        set_actnorm_identity(self.flow)  # stable initialization

    # ----- helpers -----
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

    def _token_embed(self, t_rec, ua, ur, log_dt, pos_ratio = None):
        if pos_ratio is None:
            return self.input_projection(torch.cat([t_rec, ua, ur, log_dt], dim=1))
        else:
            return self.input_projection(torch.cat([t_rec, ua, ur, log_dt, pos_ratio], dim=1))

    # ----- dequantization helper -----
    def dequantize(self, a_idx: torch.Tensor, r_idx: torch.Tensor):
        """
        Dirichlet-based dequantization + (I)LR + optional rotation.
        Inputs:
            a_idx, r_idx: (B,) integer indices (may include PAD indices).
        Returns:
            ua_for_flow, ur_for_flow : (B, A-1)/(B, R-1) transformed (rotated) coords for flow
            log_q : (B,) variational posterior log-density of sampled simplices
            logdet : (B,) log|Jacobian| from simplex -> (I)LR
            valid : (B,) bool mask (both activity & resource not PAD)
        """
        ya, log_q_a, valid_a, _ = dirichlet_dequantize_with_pad(
            a_idx,
            K=self.A,
            pad_idx=self.pad_activity_idx,
            alpha_main=self.alpha_main,
            alpha_noise=self.alpha_noise,
            pad_policy="uniform",
            alpha_pad_main=200.0,
            alpha_pad_noise=1.0,
        )
        yr, log_q_r, valid_r, _ = dirichlet_dequantize_with_pad(
            r_idx,
            K=self.R,
            pad_idx=self.pad_resource_idx,
            alpha_main=self.alpha_main,
            alpha_noise=self.alpha_noise,
            pad_policy="uniform",
            alpha_pad_main=200.0,
            alpha_pad_noise=1.0,
        )
        valid = valid_a & valid_r
        if self.use_ilr:
            ua = ilr(ya, self.ilr_basis_act)
            ur = ilr(yr, self.ilr_basis_res)
            logdet = logdet_jac_ilr(ya) + logdet_jac_ilr(yr)
        else:
            ua = alr(ya)
            ur = alr(yr)
            logdet = logdet_jac_alr(ya) + logdet_jac_alr(yr)
        ua_for_flow = self.pre_rot_a(ua) if self.use_pre_rotation else ua
        ur_for_flow = self.pre_rot_r(ur) if self.use_pre_rotation else ur
        log_q = log_q_a + log_q_r
        return ua_for_flow, ur_for_flow, log_q, logdet, valid

    def inverse_dequantize(self, ua_flow: torch.Tensor, ur_flow: torch.Tensor, log_dt: torch.Tensor):
        """
        Inverse of (rotation + (I)LR) used after sampling / flow log_prob step.
        Inputs:
            ua_flow, ur_flow: (B, A-1)/(B, R-1) in flow latent space (possibly rotated)
            log_dt: (B,1) time component (log space if use_logspace_time)
        Returns:
            act_logits: (B,A) log probs (stable)
            res_logits: (B,R) log probs
            dt_linear : (B,1) positive time in original domain
            ya_prob   : (B,A) probabilities (simplex) for activities
            yr_prob   : (B,R) probabilities (simplex) for resources
        """
        # Undo learned rotation
        ua_logits = self.pre_rot_a.inverse(ua_flow) if self.use_pre_rotation else ua_flow
        ur_logits = self.pre_rot_r.inverse(ur_flow) if self.use_pre_rotation else ur_flow
        # Map back to simplex
        if self.use_ilr:
            ya_prob = inv_ilr(ua_logits, self.ilr_basis_act)
            yr_prob = inv_ilr(ur_logits, self.ilr_basis_res)
        else:
            ya_prob = inv_alr(ua_logits)
            yr_prob = inv_alr(ur_logits)
        act_logits = torch.log(ya_prob.clamp_min(1e-8))
        res_logits = torch.log(yr_prob.clamp_min(1e-8))
        dt_linear = torch.exp(log_dt) if self.use_logspace_time else log_dt
        dt_linear = dt_linear.clamp_min(0.0)
        return act_logits, res_logits, dt_linear, ya_prob, yr_prob

    # ----- forward -----
    def forward(self, z, c=None, tuple_inputs=None, tf_ratio=None, use_beam_search: bool = False, beam_size: int = 1, beam_max_len: int = None):
        """
        If tuple_inputs is provided (acts_gt, dts_gt, ress_gt), runs teacher-forced training.
        Otherwise, runs autoregressive inference.

        Returns:
            attrs_rec (dict),
            acts_rec  (B,T,A)   logits = log probs,
            ts_rec    (B,T,1),
            ress_rec  (B,T,R)   logits = log probs,
            log_px    (B,)      sum over time of joint log-likelihood bound
        """
        eps = 1e-8
        B = z.size(0)

        # --- Latent fusion (z,c) -> trace-level context t_rec ---
        if self.is_conditional:
            z = torch.cat([z, c], dim=1)
        t_rec = self.dropout(F.relu(self.z2t(z)))

        # --- Trace attributes (unrelated to flow objective) ---
        attrs_rec = {}
        if self.has_trace_attributes:
            for trace_attr in self.trace_attributes:
                out = self.t2attr[trace_attr["name"]](t_rec)
                if trace_attr["type"] == "categorical":
                    attrs_rec[trace_attr["name"]] = F.softmax(out, dim=1)
                else:
                    attrs_rec[trace_attr["name"]] = torch.sigmoid(out)

        act_logits_list, res_logits_list, ts_list, log_terms = [], [], [], []

        # ---------- TRAINING: Pure teacher forcing (with padding-aware dequantization) ----------
        if tuple_inputs is not None:
            acts_gt, dts_gt, ress_gt = tuple_inputs     # (B,T), (B,T), (B,T)
            T = self.Tmax

            pos_ratio_start = torch.zeros(B, 1, device=self.device) if self.use_pos_ratio else None
            # Start token (position 0) has zeros for per-event parts
            start = self._token_embed(
                t_rec,
                torch.zeros(B, self.ua_dim, device=self.device),
                torch.zeros(B, self.ur_dim, device=self.device),
                torch.zeros(B, 1, device=self.device),
                pos_ratio_start
            ).unsqueeze(1)  # (B,1,cf_dim)

            # Padding masks for attention and for flow loss
            pad_act_mask = (acts_gt != self.pad_activity_idx)                   # (B,T) True = real, False = PAD
            start_mask   = torch.ones((B, 1), dtype=torch.bool, device=self.device)
            pad_full     = torch.cat([start_mask, pad_act_mask], dim=1)         # (B,T+1)

            # We will cache dequantized pieces per step to reuse after the Transformer pass
            ua_list, ur_list, log_dt_list, valid_list, logq_list, logdet_list = [], [], [], [], [], []

            tokens = [start]
            for t in range(T):
                # Ground-truth categorical ids and time
                a_idx = acts_gt[:, t]
                r_idx = ress_gt[:, t]
                dt    = dts_gt[:, t].unsqueeze(1)
                log_dt = torch.log(dt.clamp_min(eps)) if self.use_logspace_time else dt

                # # --- Padding-aware Dirichlet dequantization for (a,r) ---
                # ya, log_q_a, valid_a, _ = dirichlet_dequantize_with_pad(
                #     a_idx, K=self.A, pad_idx=self.pad_activity_idx,
                #     alpha_main=self.alpha_main, alpha_noise=self.alpha_noise,
                #     pad_policy="uniform", alpha_pad_main=200.0, alpha_pad_noise=1.0
                # )
                # yr, log_q_r, valid_r, _ = dirichlet_dequantize_with_pad(
                #     r_idx, K=self.R, pad_idx=self.pad_resource_idx,
                #     alpha_main=self.alpha_main, alpha_noise=self.alpha_noise,
                #     pad_policy="uniform", alpha_pad_main=200.0, alpha_pad_noise=1.0
                # )
                # valid = (valid_a & valid_r)  # (B,) contribute to flow only if both are non-PAD

                # # Map to R^{K-1} for the flow token; record Jacobian term for ALR
                # ua = ilr(ya, self.ilr_basis_act) if self.use_ilr else alr(ya)                    # (B, A-1)
                # ur = ilr(yr, self.ilr_basis_res) if self.use_ilr else alr(yr)                   # (B, R-1)
                # # Apply learned rotation if enabled
                # ua_for_flow = self.pre_rot_a(ua) if self.use_pre_rotation else ua
                # ur_for_flow = self.pre_rot_r(ur) if self.use_pre_rotation else ur
                # logdet = logdet_jac_ilr(ya) + logdet_jac_ilr(yr)   # (B,)
                # log_q  = log_q_a + log_q_r                         # (B,)
                
                ua_for_flow, ur_for_flow, log_q, logdet, valid = self.dequantize(a_idx, r_idx)

                if self.use_pos_ratio:
                    pos_ratio_t = torch.full((B, 1), float(t) / float(self.Tmax), device=self.device)
                else:
                    pos_ratio_t = None
                # Build token for Transformer (conditioner)
                token = self._token_embed(t_rec, ua_for_flow, ur_for_flow, log_dt, pos_ratio_t)  
                tokens.append(token.unsqueeze(1))

                # Cache for likelihood computation after context is formed
                ua_list.append(ua_for_flow)
                ur_list.append(ur_for_flow)
                log_dt_list.append(log_dt)
                valid_list.append(valid)
                logq_list.append(log_q)
                logdet_list.append(logdet)

            # Transformer over the full teacher-forced prefix (+ start)
            seq = torch.cat(tokens, dim=1)                           # (B, T+1, cf_dim)
            seq = seq + self.positional_encoding[:seq.size(1), :]    # add positions

           # CHANGE: additive causal mask slice
            attn_mask = self.causal_mask_add[:seq.size(1), :seq.size(1)]

            # CHANGE: key padding mask (True => IGNORE)
            key_padding_mask = ~pad_full[:, :seq.size(1)].bool()         # (B,T+1) bool

            # FiLM context: ONLY pass global trace embedding t_rec (shape [B, t_dim])
            # (Remove earlier concatenation logic; no need for torch.cat when using a single tensor.)
            ctx_film = t_rec if self.use_film else None

            H = self.transformer_encoder(
                seq,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                ctx=ctx_film
            )  # (B, T+1, cf_dim)

            # Per-step joint likelihood bound + logits from dequantized simplices
            for t in range(T):
                h_t   = H[:, t, :]                                   # (B, cf_dim) context for step t
                cond  = torch.cat([h_t, t_rec], dim=1)               # include global t_rec

                ua = ua_list[t]; ur = ur_list[t]; log_dt = log_dt_list[t]
                v  = torch.cat([ua, ur, log_dt], dim=1)              # (B, D)
                

                # Variational dequantization bound terms and PAD masking
                log_q   = logq_list[t]                               # (B,)
                logdet  = logdet_list[t]                             # (B,)
                valid   = valid_list[t]                     # (B,)
                
                if valid.any():
                    v_valid    = v[valid]
                    cond_valid = cond[valid]
                    log_p_valid = self.flow.log_prob(v_valid, context=cond_valid)   # (N_valid,)
                    # scatter back to (B,)
                    log_p = torch.zeros(B, device=v.device)
                    log_p[valid] = log_p_valid
                else:
                    # No valid rows at this step (all PAD): contribute zero
                    log_p = torch.zeros(B, device=v.device)
                
                
                joint_logp_bound_t = (log_p - log_q - logdet) * valid.float() 
                log_terms.append(joint_logp_bound_t)

                # Activity/resource logits as log-probs from the dequantized simplices
                # (These are stable logits to use with CE if desired)
                # Inverse rotation for logits if enabled
                ua_logits = self.pre_rot_a.inverse(ua) if self.use_pre_rotation else ua
                ur_logits = self.pre_rot_r.inverse(ur) if self.use_pre_rotation else ur
                ya = inv_ilr(ua_logits, self.ilr_basis_act) if self.use_ilr else inv_alr(ua_logits)
                yr = inv_ilr(ur_logits, self.ilr_basis_res) if self.use_ilr else inv_alr(ur_logits)
                act_logits_list.append(torch.log(ya.clamp_min(1e-8)).unsqueeze(1))
                res_logits_list.append(torch.log(yr.clamp_min(1e-8)).unsqueeze(1))
                # Return times in original (linear) space
                dt_linear = torch.exp(log_dt) if self.use_logspace_time else log_dt
                ts_list.append(dt_linear.unsqueeze(1))

        # ---------- INFERENCE: Autoregressive sampling from the flow ----------
        else:
            # ========================
            # INFERENCE (Sampling or Beam Search)
            # ========================
            # NOTE: For beam search we only score by the flow log-prob (approximate)
            # and ignore attribute heads. This keeps implementation lightweight.
            beam_max_len = beam_max_len or self.Tmax

            if not use_beam_search or beam_size <= 1:
                # -------- Original greedy sampling path (unchanged aside from comments) --------
                seq = torch.zeros(B, 1, self.cf_dim, device=self.device)
                if self.use_pos_ratio:
                    pos_ratio_start = torch.zeros(B, 1, device=self.device)
                else:
                    pos_ratio_start = None
                start = self._token_embed(
                    t_rec,
                    torch.zeros(B, self.ua_dim, device=self.device),
                    torch.zeros(B, self.ur_dim, device=self.device),
                    torch.zeros(B, 1, device=self.device),
                    pos_ratio_start
                )
                seq[:, 0, :] = start

                for _ in range(self.Tmax):  # autoregressive loop
                    seq_pos = seq + self.positional_encoding[:seq.size(1), :]
                    attn_mask = self.causal_mask_add[:seq.size(1), :seq.size(1)]
                    key_padding_mask = None
                    ctx_film = t_rec if self.use_film else None
                    H = self.transformer_encoder(
                        seq_pos,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                        ctx=ctx_film
                    )
                    h_t = H[:, -1, :]
                    cond = torch.cat([h_t, t_rec], dim=1)
                    z_base = torch.randn(B, self.ua_dim + self.ur_dim + 1, device=self.device)
                    v_samp, _ = self.flow._transform.inverse(z_base, context=cond)
                    ua, ur, log_dt = torch.split(v_samp, [self.ua_dim, self.ur_dim, 1], dim=1)
                    act_l, res_l, dt_lin, _, _ = self.inverse_dequantize(ua, ur, log_dt)
                    act_logits_list.append(act_l.unsqueeze(1))
                    res_logits_list.append(res_l.unsqueeze(1))
                    ts_list.append(dt_lin.unsqueeze(1))
                    t_next = seq.size(1) - 1
                    if self.use_pos_ratio:
                        pos_ratio_next = torch.full((B, 1), float(t_next) / float(self.Tmax), device=self.device)
                    else:
                        pos_ratio_next = None
                    token_next = self._token_embed(t_rec, ua, ur, log_dt, pos_ratio_next)
                    seq = torch.cat([seq, token_next.unsqueeze(1)], dim=1)
                log_terms = [torch.zeros(B, device=self.device)] * self.Tmax
            else:
                # ==============================================================
                # OPTIMIZED BEAM SEARCH (vectorized, best-beam only)
                # ==============================================================
                # DESIGN GOALS:
                #   * Keep training & greedy paths untouched.
                #   * Single vectorized Transformer pass per time step over all beams.
                #   * Use the flow's log_prob as an (approximate) additive beam score.
                #   * Sample K (= beam_size) candidate continuations per beam step, then
                #     select the top 'beam_size' among (beam_size * K) candidates.
                #   * Do NOT store full intermediate beam trees; retain only the final
                #     best beam path (lowest memory).
                #   * No early stopping (kept consistent with greedy path as requested).
                #
                # TERMINOLOGY:
                #   B  : batch size
                #   beam_size = number of beams retained after each expansion
                #   proposals_per_beam = number of stochastic samples per retained beam
                #   Tmax / beam_max_len : maximum decoding length
                #
                # SHAPES (core tensors during expansion step):
                #   tokens            : (B, beam, cur_len, cf_dim)
                #   h_last            : (B, beam, cf_dim)   last hidden state per beam
                #   cond_rep_flat     : (B * beam * K, ctx_dim)   flow conditioning
                #   v_samp             : (B * beam * K, ua_dim + ur_dim + 1)
                #   ua_all, ur_all     : (B, beam, K, ua_dim / ur_dim)
                #   ya_all, yr_all     : (B, beam, K, A / R)   simplex (activities/resources)
                #   log_p_all          : (B, beam, K) flow log-density per candidate
                #   scores             : (B, beam) cumulative beam scores
                #   scores_expanded    : (B, beam, K) additive expansion scores
                #   cand_scores        : (B, beam * K) flattened for top-k
                #
                # FINAL EXTRACTION:
                #   We store per-step chosen distributions (acts_steps, ress_steps, ts_steps)
                #   each shaped (B, beam, 1, vocab_or_1). After all steps, we select the
                #   best beam index per batch and concatenate across time → (B, T, dim).
                # ==============================================================

                proposals_per_beam = beam_size          # K expansions per current beam
                Dv = self.ua_dim + self.ur_dim + 1      # total flow latent dimension per event
                # (We intentionally ignore any EOT index for consistency with greedy path.)

                # ---- Initialize start token for every beam ----
                if self.use_pos_ratio:
                    pos_ratio_start = torch.zeros(B, 1, device=self.device)  # relative positional ratio (t/T)
                else:
                    pos_ratio_start = None

                # start_tok: (B, cf_dim) = embedding of an "empty" first event (all zeros for event parts)
                start_tok = self._token_embed(
                    t_rec,
                    torch.zeros(B, self.ua_dim, device=self.device),   # ua placeholder
                    torch.zeros(B, self.ur_dim, device=self.device),   # ur placeholder
                    torch.zeros(B, 1, device=self.device),             # log_dt placeholder
                    pos_ratio_start
                )

                # tokens: replicate start token across beams → shape (B, beam, time=1, cf_dim)
                tokens = start_tok.unsqueeze(1).unsqueeze(2).repeat(1, beam_size, 1, 1)

                # scores: running cumulative log score per beam (initialized at 0)
                scores = torch.zeros(B, beam_size, device=self.device)

                # Containers to collect per-step chosen (log) distributions & times for each beam
                # We store them at each step so we can reconstruct the path for the best beam only.
                acts_steps, ress_steps, ts_steps = [], [], []

                # ------------- Time-step expansion loop -------------
                # Each iteration expands all beams in parallel.
                for t_step in range(min(beam_max_len, self.Tmax)):
                    # Current decoded length (including the start token position)
                    cur_len = tokens.size(2)

                    # Flatten (B, beam, cur_len, cf_dim) → (B * beam, cur_len, cf_dim)
                    # so the Transformer processes each beam as an independent sequence.
                    seq_flat = tokens.view(B * beam_size, cur_len, self.cf_dim)

                    # Add positional encodings (broadcast over B*beam)
                    seq_flat = seq_flat + self.positional_encoding[:cur_len, :]

                    # Transformer forward for all beams in one pass
                    H = self.transformer_encoder(
                        seq_flat,
                        attn_mask=self.causal_mask_add[:cur_len, :cur_len],   # causal (additive) mask
                        key_padding_mask=None,                                # no padding in inference
                        ctx=(t_rec.repeat_interleave(beam_size, 0) if self.use_film else None)
                    )
                    # Take the last hidden state per beam sequence (next-event context)
                    h_last = H[:, -1, :].view(B, beam_size, -1)

                    # Condition for flow: concat per-beam last hidden + broadcast trace embedding t_rec
                    t_rec_exp = t_rec.unsqueeze(1).expand(B, beam_size, self.t_dim)
                    cond = torch.cat([h_last, t_rec_exp], dim=-1)  # (B, beam, ctx_dim)

                    # ---- Generate K stochastic proposals per existing beam (vectorized) ----
                    cond_rep = cond.unsqueeze(2).repeat(1, 1, proposals_per_beam, 1)  # (B, beam, K, ctx_dim)
                    # Flatten for the flow: (B*beam*K, ctx_dim)
                    cond_rep_flat = cond_rep.view(B * beam_size * proposals_per_beam, -1)

                    # Sample base noise for all proposals at once
                    z_base = torch.randn(B * beam_size * proposals_per_beam, Dv, device=self.device)

                    # Invert flow transform → samples in event latent space (ua, ur, log_dt)
                    v_samp, _ = self.flow._transform.inverse(z_base, context=cond_rep_flat)

                    # Split latent tuple into activity part, resource part, time part
                    ua_all, ur_all, log_dt_all = torch.split(v_samp, [self.ua_dim, self.ur_dim, 1], dim=1)

                    # Flow log probability (density) per sampled candidate; shape -> (B, beam, K)
                    log_p_all = self.flow.log_prob(v_samp, context=cond_rep_flat)\
                                   .view(B, beam_size, proposals_per_beam)

                    # Reshape each latent slice back to (B, beam, K, dim)
                    ua_all = ua_all.view(B, beam_size, proposals_per_beam, self.ua_dim)
                    ur_all = ur_all.view(B, beam_size, proposals_per_beam, self.ur_dim)
                    log_dt_all = log_dt_all.view(B, beam_size, proposals_per_beam, 1)

                    # Note: we defer inverse mapping (rotation + simplex) to only the
                    # selected candidates via inverse_dequantize for efficiency.

                    # ---- Expand beam scores: add flow log_prob for each proposal ----
                    scores_expanded = scores.unsqueeze(-1) + log_p_all  # (B, beam, K)

                    # Flatten candidate matrix per batch element: (B, beam*K)
                    cand_scores = scores_expanded.view(B, beam_size * proposals_per_beam)

                    # Select top 'beam_size' candidates among all (beam_size*K) expansions
                    top_scores, top_indices = torch.topk(cand_scores, k=beam_size, dim=-1)
                    # Derive originating beam index and proposal index within that beam
                    parent_beam = top_indices // proposals_per_beam  # (B, beam)
                    prop_index  = top_indices % proposals_per_beam   # (B, beam)

                    # ---- Gather chosen candidate tensors (vectorized per batch) ----
                    batch_idx = torch.arange(B, device=self.device).unsqueeze(1)
                    ua_chosen = ua_all[batch_idx, parent_beam, prop_index]      # (B, beam, ua_dim)
                    ur_chosen = ur_all[batch_idx, parent_beam, prop_index]      # (B, beam, ur_dim)
                    log_dt_chosen = log_dt_all[batch_idx, parent_beam, prop_index]  # (B, beam, 1)

                    # Flatten beams to apply inverse_dequantize, then reshape back
                    flat_ua = ua_chosen.view(B * beam_size, self.ua_dim)
                    flat_ur = ur_chosen.view(B * beam_size, self.ur_dim)
                    flat_log_dt = log_dt_chosen.view(B * beam_size, 1)
                    act_l_flat, res_l_flat, dt_lin_flat, _, _ = self.inverse_dequantize(
                        flat_ua, flat_ur, flat_log_dt
                    )
                    act_l = act_l_flat.view(B, beam_size, self.A)
                    res_l = res_l_flat.view(B, beam_size, self.R)
                    dt_lin = dt_lin_flat.view(B, beam_size, 1)

                    acts_steps.append(act_l.unsqueeze(2))         # (B, beam, 1, A)
                    ress_steps.append(res_l.unsqueeze(2))         # (B, beam, 1, R)
                    ts_steps.append(dt_lin.unsqueeze(2))          # (B, beam, 1, 1)

                    # Prepare token embedding for next step context conditioning:
                    ua_next = ua_all[batch_idx, parent_beam, prop_index]
                    ur_next = ur_all[batch_idx, parent_beam, prop_index]
                    log_dt_next = log_dt_all[batch_idx, parent_beam, prop_index]

                    if self.use_pos_ratio:
                        pos_ratio_next = torch.full(
                            (B, beam_size, 1),
                            float(t_step) / float(self.Tmax),
                            device=self.device
                        )
                    else:
                        pos_ratio_next = None

                    # _token_embed expects flattened (B*beam, dim) arguments
                    tok_next = self._token_embed(
                        t_rec.unsqueeze(1).expand(-1, beam_size, -1).reshape(B * beam_size, -1),
                        ua_next.reshape(B * beam_size, -1),
                        ur_next.reshape(B * beam_size, -1),
                        log_dt_next.reshape(B * beam_size, -1),
                        pos_ratio_next.reshape(B * beam_size, -1) if self.use_pos_ratio else None
                    ).view(B, beam_size, -1)  # (B, beam, cf_dim)

                    # Append next token to every beam's sequence: increase time dimension by 1
                    tokens = torch.cat([tokens, tok_next.unsqueeze(2)], dim=2)

                    # Update cumulative beam scores to top_scores for next iteration
                    scores = top_scores

                # ------------- Final beam selection -------------
                # Identify best beam index per batch element (highest cumulative log score)
                best_beam = scores.argmax(dim=1)  # (B,)

                # Reconstruct chosen beam path across ALL generated steps (Tgen = #expansions)
                act_logits_list, res_logits_list, ts_list = [], [], []
                Tgen = len(acts_steps)

                for t in range(Tgen):
                    # acts_steps[t]: (B, beam, 1, A) → select beam dimension
                    act_logits_list.append(
                        acts_steps[t][torch.arange(B), best_beam]  # (B,1,A)
                    )
                    res_logits_list.append(
                        ress_steps[t][torch.arange(B), best_beam]  # (B,1,R)
                    )
                    ts_list.append(
                        ts_steps[t][torch.arange(B), best_beam]    # (B,1,1)
                    )

                # Concatenate along time: (B, Tgen, A/R/1)
                acts_rec = torch.cat(act_logits_list, dim=1) if Tgen else torch.zeros(B, 0, self.A, device=self.device)
                ress_rec = torch.cat(res_logits_list, dim=1) if Tgen else torch.zeros(B, 0, self.R, device=self.device)
                ts_rec = torch.cat(ts_list, dim=1) if Tgen else torch.zeros(B, 0, 1, device=self.device)

                # Right-pad to fixed Tmax length for downstream tensor shape expectations
                if acts_rec.size(1) < self.Tmax:
                    pad_T = self.Tmax - acts_rec.size(1)
                    acts_rec = torch.cat([acts_rec, torch.zeros(B, pad_T, self.A, device=self.device)], dim=1)
                    ress_rec = torch.cat([ress_rec, torch.zeros(B, pad_T, self.R, device=self.device)], dim=1)
                    ts_rec = torch.cat([ts_rec, torch.zeros(B, pad_T, 1, device=self.device)], dim=1)

                # Normalize output format to match greedy path (lists for later stacking)
                act_logits_list = [acts_rec]
                res_logits_list = [ress_rec]
                ts_list = [ts_rec]
                # No likelihood accumulation in inference mode (dummy zeros)
                log_terms = [torch.zeros(B, device=self.device)] * self.Tmax

        # --- Stack outputs ---
        acts_rec = torch.cat(act_logits_list, dim=1)  # (B,T,A) logits
        ress_rec = torch.cat(res_logits_list, dim=1)  # (B,T,R) logits
        ts_rec   = torch.cat(ts_list,        dim=1)   # (B,T,1)
        log_px   = torch.stack(log_terms,    dim=1).sum(dim=1)  # (B,)

        return attrs_rec, acts_rec, ts_rec, ress_rec, log_px

