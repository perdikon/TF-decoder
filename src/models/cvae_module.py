from tracemalloc import start
from src.cvae_evaluation_pipeline import CVAEEvaluationPipeline
import torch
from torch import nn
import torch.nn.functional as F
import math

from lightning import LightningModule

from typing import Tuple, Dict

from collections import Counter, OrderedDict, defaultdict
from torch.autograd.function import Function
import torch


class CVAELitModule(LightningModule):
    def __init__(self, vae, **kwargs):
        """
        The Lightning Module accepts a configuration and dataset_info for setting up the model.
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["vae"])

        self._vae_builder = vae  # ← a functools.partial
        self.model = None
        self.trace_attributes = None  # Placeholder for trace attributes

        self._cat_attr_loss = nn.BCELoss(reduction="sum")
        self._num_attr_loss = nn.MSELoss(reduction="sum")
        self._cf_loss = nn.BCELoss(reduction="sum")
        self._ts_loss = nn.MSELoss(reduction="sum")
        self._res_loss = nn.BCELoss(reduction="sum")

        self._kl_schedule = [0]

        self.MMD = None

    def configure_model(self) -> None:
        self.dataset_info = self.trainer.datamodule.dataset_info
        
        dataset = self.trainer.datamodule.data_train if self.trainer.training else self.trainer.datamodule.data_test
        padding_resource = dataset.PADDING_RESOURCE
        padding_activity = dataset.PADDING_ACTIVITY
        padding_resource_idx = dataset.resource2n.get(padding_resource, 0)
        padding_activity_idx = dataset.activity2n.get(padding_activity, 0)
        self.model = self._vae_builder(
            trace_attributes=self.dataset_info.trace_attributes,
            num_activities=self.dataset_info.num_activities,
            num_resources=self.dataset_info.num_resources,
            max_trace_length=self.dataset_info.max_trace_length,
            device=self.device,
            pad_activity_idx=padding_activity_idx,
            pad_resource_idx=padding_resource_idx,
        )

        
        from src.models.components.cvae.transformer_flow_decoder_new import (
            TransformerFlowDecoderSimplex,
        )

        self.is_flow_decoder = isinstance(self.model.decoder, TransformerFlowDecoderSimplex)

        # self.MMD = MMD(dm=self.trainer.datamodule.full_dataset, model=self.model)

    def setup(self, stage=None):
        super().setup(stage)
        if stage in ["validate", "train", None]:
            trainer = self.trainer
            if self.is_flow_decoder:
                dataloader = self.trainer.datamodule.train_dataloader()
                batch = next(iter(dataloader))
                x, c = batch[0], batch[1]
                acts, dts, ress = x[1], x[2], x[3]
                tuple_vec = torch.cat(
                    [
                        self.model.decoder.encoder_act2e(acts[:, 0]),
                        self.model.decoder.encoder_res2e(ress[:, 0]),
                        torch.log(dts[:, 0:1] + 1e-8),
                    ],
                    dim=1,
                )
                cond_vec = torch.zeros(
                    (acts.size(0), self.model.decoder.context_dim), device=acts.device
                )
                self.model.decoder.flow._transform(tuple_vec, context=cond_vec)

    def on_fit_start(self) -> None:

        n_iter = self.trainer.max_epochs
        h = self.hparams.kl_cycle

        L = torch.ones(n_iter, dtype=torch.float32, device=self.device) * h.stop
        period = n_iter / h.n_cycles
        step = (h.stop - h.start) / (period * h.ratio)  # linear increment

        for c in range(h.n_cycles):
            v, i = h.start, 0
            while v <= h.stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1

        self._kl_schedule = L
        
    # def on_train_start(self):
    #     self.train()
    #     batch = next(iter(self.trainer.datamodule.train_dataloader()))
    #     _warmup_actnorm_from_single_batch(self.model, batch, pad_policy="uniform")
    #     return super().on_train_start()

    def forward(self, x, c=None):
        # Delegate to the VAE model; Lightning’s forward is used for inference.
        return self.model(x, c)

    def training_step(self, batch, batch_idx):
        kl_w = self.get_kl_weight(self.current_epoch)
        # kl_w = min(self.current_epoch / 200, 0.5)
        if self.is_flow_decoder:
            return self.flow_shared_step_simplex(batch=batch, stage="train", kl_w=kl_w)
        else:
            return self._shared_step(batch=batch, stage="train", kl_w=kl_w)

    def validation_step(self, batch, batch_idx):

        kl_w = self.get_kl_weight(self.current_epoch)
        # kl_w = min(self.current_epoch / 200, 0.5)
        if self.is_flow_decoder:
            return self.flow_shared_step_simplex(batch=batch, stage="val", kl_w=kl_w)
        else:
            return self._shared_step(batch=batch, stage="val", kl_w=kl_w)

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        self.model.eval()

        test_cfg = self.trainer.cfg.get("test_cfg", {})
        
        if test_cfg:
            # Create evaluation pipeline
            pipeline = CVAEEvaluationPipeline(
                model=self.model, datamodule=self.trainer.datamodule, config=test_cfg
            )

            # Build output path
            experiment_name = test_cfg.get("output", {}).get(
                "experiment_name", "testing_default"
            )
            
            #cfg.paths.output_dir
            output_path = f"{self.trainer.cfg.paths.output_dir}/{experiment_name}"

            # Run evaluation
            results = pipeline.run_full_evaluation(output_path)
            
            return results

        return 0

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def get_kl_weight(self, epoch):
        if self.hparams.flow_warmup_epochs is None:
            return float(self._kl_schedule[epoch])
        else:
            if epoch < self.hparams.flow_warmup_epochs:
                return float(self._kl_schedule[epoch])  # Stage-1 as before
            else:
                return 1.0  # or 1.0 if you still want it logged

    def on_after_backward(self):
        if self.is_flow_decoder:
            if self.current_epoch % 1 == 0:  # Log every epoch
                grad_norm = (
                    sum(
                        p.grad.norm().item() ** 2
                        for p in self.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                flow_params = (
                    sum(p.norm().item() ** 2 for p in self.model.decoder.flow.parameters())
                    ** 0.5
                )
                self.log(
                    "grad_norm", grad_norm, on_step=False, on_epoch=True, prog_bar=True
                )
                self.log(
                    "flow_params", flow_params, on_step=False, on_epoch=True, prog_bar=True
                )

    def _new_shared_step(self, batch, stage: str, kl_w: float) -> torch.Tensor:
        x, y = batch

        x_rec, mean, logvar = self(x, y)

        self.log_dict(
            {
                f"{stage}/mean": mean.mean(),
                f"{stage}/logvar": logvar.mean(),
            },
            on_epoch=True,
        )

        rec_loss, cat_attrs = self.reconstruction_loss_fn_2(x_rec, x)
        # rec_loss_bce, cat_attrs_bce = self._reconstruction_loss_bce(x_rec, x)

        kl_loss, dimwise_kl = self.kl_divergence_loss_fn2(mean, logvar)
        loss = rec_loss + kl_w * kl_loss  

        # consistent naming -> easy filtering in TensorBoard/W&B
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/rec_loss": rec_loss,
                f"{stage}/kl_loss": kl_loss,
                "kl_weight": kl_w,
            },
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            {
                f"{stage}/rec/cat_attrs_loss": cat_attrs["cat_attrs"],
                f"{stage}/rec/num_attrs_loss": cat_attrs["num_attrs"],
                f"{stage}/rec/cf_loss": cat_attrs["act"],
                f"{stage}/rec/ts_loss": cat_attrs["ts"],
                f"{stage}/rec/res_loss": cat_attrs["res"],
            },
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def kl_divergence_loss_fn_flow(self, mean, var, free_bits=0.0):
        """KL divergence with free bits to prevent posterior collapse"""
        var = torch.clamp(var, min=1e-9)

        kl_loss = -0.5 * torch.sum(1 + torch.log(var**2) - mean**2 - var**2)

        # Apply free bits - ensure minimum KL per dimension
        latent_dim = mean.size(1)
        min_kl = free_bits * latent_dim
        return torch.max(kl_loss, torch.tensor(min_kl, device=kl_loss.device))

    def kl_divergence_loss_fn_flow_2(self, mean, logvar, free_bits=0.03):
        # KL per dimension per batch element: [B, latent_dim]
        dimwise = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())

        # Apply free bits per dimension per batch element
        if free_bits > 0.0:
            dimwise = torch.clamp(dimwise, min=free_bits)

        # Sum across dimensions, mean across batch
        return dimwise.sum(dim=1).mean(), dimwise  # Better than sum over everything

    def reconstruction_loss_fn_2(self, x_rec, x):
        """
        Updated reconstruction loss function with magnitudes matching flow_reconstruction_loss_fn.
        Uses mean reduction and returns losses in the same format as flow_reconstruction_loss_fn.
        
        Args:
            x_rec: Tuple of (attrs_rec, acts_rec, ts_rec, ress_rec) from model output
            x: Tuple of (attrs, acts, ts, ress) ground truth
                
        Returns:
            total_loss: Combined reconstruction loss
            components: Dictionary with individual loss components (for compatibility with flow_shared_step)
        """
        attrs_rec, acts_rec, ts_rec, ress_rec = x_rec
        attrs, acts_gt, ts_gt, ress_gt = x
        
        dm = self.trainer.datamodule.data_train
        
        # Initialize losses as dictionary to match flow_reconstruction_loss_fn format
        losses = {
            "cat_attrs": torch.tensor(0.0, device=self.device),
            "num_attrs": torch.tensor(0.0, device=self.device),
            "act": torch.tensor(0.0, device=self.device),
            "ts": torch.tensor(0.0, device=self.device),
            "res": torch.tensor(0.0, device=self.device)
        }
        
        # ===== 1. Process attributes =====
        if attrs and attrs_rec:
            for name, val in attrs.items():
                if name in dm.s2i:  # Categorical attribute
                    target = F.one_hot(
                        val.to(torch.int64),
                        num_classes=len(dm.s2i[name]),
                    ).float()
                    # Changed: Using mean reduction
                    attr_loss = F.binary_cross_entropy_with_logits(
                        attrs_rec[name], target, reduction="sum"
                    )
                    losses["cat_attrs"] += attr_loss
                else:  # Numerical attribute
                    pred = attrs_rec[name].squeeze()
                    target = val.float().squeeze()
                    # Changed: Using mean reduction
                    attr_loss = F.mse_loss(pred, target, reduction="sum")
                    losses["num_attrs"] += attr_loss
        
        # ===== 2. Activities =====
        pad_act = dm.activity2n.get(dm.PADDING_ACTIVITY, 0)
        
        # Changed: Using mean reduction and no scaling by L
        B, L, V_act = acts_rec.shape
        losses["act"] = F.cross_entropy(
            acts_rec.reshape(-1, V_act),    # [B*L, V_act]
            acts_gt.reshape(-1).long(),     # [B*L]
            ignore_index=pad_act,           # Ignore padding tokens
            reduction="sum"                # Changed to mean
        )
        
        # ===== 3. Timestamps =====
        
        # ts
        # ts_loss_fn = nn.MSELoss(reduction="sum")
        # lens = acts.argmax(dim=2).argmax(dim=1)
        # # use lens and compute loss only for the first lens elements
        # for i in range(len(lens)):
        #     if lens[i] < ts_rec.size(1):
        #         ts_rec[i, lens[i] :] = 0.0

        # ts_loss += ts_loss_fn(ts_rec, ts)
        
        
        # Create mask for valid timestamps (not padding)
        ts_mask = (acts_gt != pad_act).float()
        ts_rec = ts_rec.squeeze(-1) if ts_rec.dim() > 2 else ts_rec
        
        
        # Changed: Calculate mean loss over valid timesteps
        ts_squared_diff = (ts_rec - ts_gt) ** 2
        masked_ts_loss = (ts_squared_diff * ts_mask).sum()
        valid_ts_positions = ts_mask.sum().clamp(min=1.0)  # Avoid division by zero
        losses["ts"] = masked_ts_loss / valid_ts_positions
        
        # ===== 4. Resources =====
        pad_res = dm.resource2n.get(dm.PADDING_RESOURCE, 0)
        
        # Changed: Using mean reduction and no scaling by L
        B, L, V_res = ress_rec.shape
        losses["res"] = F.cross_entropy(
            ress_rec.reshape(-1, V_res),    # [B*L, V_res]
            ress_gt.reshape(-1).long(),     # [B*L]
            ignore_index=pad_res,           # Ignore padding tokens
            reduction="sum"                # Changed to mean
        )
        
        # ===== 5. Combine losses =====
        total_loss = (
            losses["cat_attrs"] + 
            losses["num_attrs"] + 
            losses["act"] + 
            losses["ts"] + 
            losses["res"]
        )
        
        # Return format matches flow_reconstruction_loss_fn
        return total_loss, losses
    
    def _shared_step(self, batch, stage: str, kl_w: float) -> torch.Tensor:
        x, y = batch

        x_rec, mean, logvar = self(x, y)

        self.log_dict(
            {
                f"{stage}/mean": mean.mean(),
                f"{stage}/logvar": logvar.mean(),
            },
            on_epoch=True,
        )
        # Check for NaN/Inf in mean and logvar
        if torch.isfinite(mean).all() and torch.isfinite(logvar).all():
            pass
        else:
            raise ValueError("NaN/Inf detected in mu or logvar")

        rec_loss, cat_attrs = self._reconstruction_loss(x_rec, x)
       
        kl_loss = self.kl_divergence_loss_fn(mean, logvar)
        loss = rec_loss + kl_w * kl_loss  # + mmd_value

        # consistent naming -> easy filtering in TensorBoard/W&B
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/rec_loss": rec_loss,
                f"{stage}/kl_loss": kl_loss,
                f"{stage}/kl_weight": kl_w,
            },
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            {
                f"{stage}/rec/cat_attrs_loss": cat_attrs["cat_attrs"],
                f"{stage}/rec/num_attrs_loss": cat_attrs["num_attrs"],
                f"{stage}/rec/cf_loss": cat_attrs["cf"],
                f"{stage}/rec/ts_loss": cat_attrs["ts"],
                f"{stage}/rec/res_loss": cat_attrs["res"],
            },
            on_epoch=True,
        )

        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    
    def _reconstruction_loss(
        self,
        x_rec: Tuple,  # (attrs_rec, acts_rec, ts_rec, ress_rec)
        x: Tuple,  # (attrs, acts, ts, ress) – as yielded by dataloader
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Re-implementation of the original `reconstruction_loss_fn`,
        but without global variables or explicit device juggling.
        """
        # split incoming tuples
        attrs_rec, acts_rec, ts_rec, ress_rec = x_rec
        attrs, acts, ts, ress = x
        

        dm = self.trainer.datamodule.data_train

        # ------------------------------------------------
        # 1) Attributes
        # ------------------------------------------------
        cat_attr_losses, num_attr_losses = [], []

        for name, val in attrs.items():
            if name in dm.s2i:  # categorical
                target = F.one_hot(
                    val.to(torch.int64),
                    num_classes=len(dm.s2i[name]),  # avoid “index ≥ num_classes”
                ).float()
                cat_attr_losses.append(self._cat_attr_loss(attrs_rec[name], target))
            else:  # numerical
                pred = attrs_rec[name].squeeze()
                num_attr_losses.append(self._num_attr_loss(pred, val.float()))

        cat_attrs_loss = (
            torch.stack(cat_attr_losses).sum()
            if cat_attr_losses
            else torch.tensor(0.0, device=self.device)
        )
        num_attrs_loss = (
            torch.stack(num_attr_losses).sum()
            if num_attr_losses
            else torch.tensor(0.0, device=self.device)
        )

        # ------------------------------------------------
        # 2) Activities (control-flow)
        # ------------------------------------------------
        pad_act = dm.activity2n[dm.PADDING_ACTIVITY]
        eot_act = dm.activity2n[dm.EOT_ACTIVITY]
        acts_idx = torch.where(
            acts == pad_act,
            eot_act,
            acts,
        )

        acts = F.one_hot(
            acts_idx.to(torch.int64),
            num_classes=len(dm.activity2n) - 1,  # derive from mapping
        ).float()
        cf_loss = self._cf_loss(acts_rec, acts)

        # ------------------------------------------------
        # 3) Timestamps
        # ------------------------------------------------
        with torch.no_grad():
            first_eot = (acts_idx == eot_act).float().argmax(dim=1)
        B, L = ts_rec.shape
        range_row = torch.arange(L, device=ts_rec.device).unsqueeze(0).expand(B, L)
        mask = range_row >= first_eot.unsqueeze(1)  # (B, L) bool

        ts_pred_mask = ts_rec.masked_fill(mask, 0.0)
        ts_loss = self._ts_loss(ts_pred_mask, ts)  # `ts` already has zeros

        # ------------------------------------------------
        # 4) Resources
        # ------------------------------------------------
        pad_res = dm.resource2n[dm.PADDING_RESOURCE]
        eot_res = dm.resource2n[dm.EOT_RESOURCE]
        ress = torch.where(
            ress == pad_res,
            eot_res,
            ress,
        )
        ress = F.one_hot(ress.to(torch.int64), num_classes=len(dm.resource2n) - 1).float()
        res_loss = self._res_loss(ress_rec, ress)

        total = cat_attrs_loss + num_attrs_loss + cf_loss + ts_loss + res_loss
        comps = {
            "cat_attrs": cat_attrs_loss,
            "num_attrs": num_attrs_loss,
            "cf": cf_loss,
            "ts": ts_loss,
            "res": res_loss,
        }
        return total, comps
    
    def _reconstruction_loss2(
        self,
        x_rec: Tuple,  # (attrs_rec, acts_rec, ts_rec, ress_rec)
        x: Tuple,  # (attrs, acts, ts, ress) – as yielded by dataloader
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Re-implementation of the original `reconstruction_loss_fn`,
        but without global variables or explicit device juggling.
        """
        # split incoming tuples
        attrs_rec, acts_rec, ts_rec, ress_rec = x_rec
        attrs, acts, ts, ress = x
        

        dm = self.trainer.datamodule.data_train

        # ------------------------------------------------
        # 1) Attributes
        # ------------------------------------------------
        cat_attr_losses, num_attr_losses = [], []

        for name, val in attrs.items():
            if name in dm.s2i:  # categorical
                target = F.one_hot(
                    val.to(torch.int64),
                    num_classes=len(dm.s2i[name]),  # avoid “index ≥ num_classes”
                ).float()
                cat_attr_losses.append(self._cat_attr_loss(attrs_rec[name], target))
            else:  # numerical
                pred = attrs_rec[name].squeeze()
                num_attr_losses.append(self._num_attr_loss(pred, val.float()))

        cat_attrs_loss = (
            torch.stack(cat_attr_losses).sum()
            if cat_attr_losses
            else torch.tensor(0.0, device=self.device)
        )
        num_attrs_loss = (
            torch.stack(num_attr_losses).sum()
            if num_attr_losses
            else torch.tensor(0.0, device=self.device)
        )

        # ------------------------------------------------
        # 2) Activities (control-flow)
        # ------------------------------------------------
        pad_act = dm.activity2n[dm.PADDING_ACTIVITY]
        eot_act = dm.activity2n[dm.EOT_ACTIVITY]
        acts_idx = torch.where(
            acts == pad_act,
            eot_act,
            acts,
        )
        
        # Compute cross-entropy loss for activities
        B, L, V_act = acts_rec.shape
        cf_loss = F.cross_entropy(
            acts_rec.reshape(-1, V_act),  # [B*L, V_act]
            acts_idx.reshape(-1).long(),   # [B*L]       
            reduction="sum"
        )


        # ------------------------------------------------
        # 3) Timestamps
        # ------------------------------------------------
        with torch.no_grad():
            first_eot = (acts_idx == eot_act).float().argmax(dim=1)
        ts_rec = ts_rec.squeeze(-1)
        B, L = ts_rec.shape
        range_row = torch.arange(L, device=ts_rec.device).unsqueeze(0).expand(B, L)
        mask = range_row >= first_eot.unsqueeze(1)  # (B, L) bool

        ts_pred_mask = ts_rec.masked_fill(mask, 0.0)
        ts_loss = self._ts_loss(ts_pred_mask, ts)  # `ts` already has zeros

        # ------------------------------------------------
        # 4) Resources
        # ------------------------------------------------
        pad_res = dm.resource2n[dm.PADDING_RESOURCE]
        eot_res = dm.resource2n[dm.EOT_RESOURCE]
        ress = torch.where(
            ress == pad_res,
            eot_res,
            ress,
        )
        B, L, V_res = ress_rec.shape
        res_loss = F.cross_entropy(
            ress_rec.reshape(-1, V_res),  # [B*L, V_res]
            ress.reshape(-1).long(),   # [B*L]
            reduction="sum"
        )

        total = cat_attrs_loss + num_attrs_loss + cf_loss + ts_loss + res_loss
        comps = {
            "cat_attrs": cat_attrs_loss,
            "num_attrs": num_attrs_loss,
            "cf": cf_loss,
            "ts": ts_loss,
            "res": res_loss,
        }
        return total, comps
    
    # def kl_divergence_loss_fn2(self, mean, logvar, free_bits=0.02):
    #     """
    #     KL divergence with same scale as flow model version.
    #     Applies free bits per dimension and uses mean reduction.
        
    #     Args:
    #         mean: Mean of the encoder distribution q(z|x)
    #         logvar: Log variance of the encoder distribution q(z|x)
    #         free_bits: Minimum KL per dimension (default: 0.02 to match config)
            
    #     Returns:
    #         kl_loss: Mean KL divergence loss
    #         dimwise_kl: Per-dimension KL for analysis
    #     """
    #     # Compute KL per dimension: -0.5 * (1 + logvar - mean² - exp(logvar))
    #     dimwise_kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        
    #     # Apply free bits per dimension
    #     if free_bits > 0.0:
    #         dimwise_kl = torch.clamp(dimwise_kl, min=free_bits)
        
    #     # Sum across dimensions, mean across batch (like flow_2 version)
    #     kl_loss = dimwise_kl.sum(dim=1).mean()
        
    #     return kl_loss, dimwise_kl

    # def _shared_step2(self, batch, stage: str, kl_w: float) -> torch.Tensor:

    #     # ---------------------------------------------------------
    #     # 1. unpack batch
    #     # ---------------------------------------------------------
    #     x_tuple = (
    #         batch["attributes"],  # dict of trace-level attrs
    #         batch["acts_padded"],  # [B,T]
    #         batch["timestamps_padded"],  # [B,T]
    #         batch["resources_padded"],  # [B,T]
    #     )
    #     y = batch["labels"]

    #     # ---------------------------------------------------------
    #     # 2. forward pass through the CVAE
    #     #    self(x,y) must now return:
    #     #       (attrs_rec, log_px_given_z), mean, logvar
    #     # ---------------------------------------------------------
    #     test = self(x_tuple, y)  # call the model with x_tuple and y
    #     (attrs_rec, log_px), mean, logvar = self(x_tuple, y)  # ★ NEW

    #     # ---------------------------------------------------------
    #     # 3. reconstruction loss  (joint NLL from the flow)
    #     # ---------------------------------------------------------
    #     rec_loss = -log_px.mean()  # minimise –log p(x|z)       ★ NEW

    #     # ---------------------------------------------------------
    #     # 4. optional: trace-level attribute losses stay BCE / CE
    #     # ---------------------------------------------------------
    #     attr_loss = 0.0
    #     # attr_loss, cat_attrs = self._reconstruction_loss_attrs(   # ★ NEW helper
    #     #     attrs_rec, batch["attributes"]
    #     # )

    #     # ---------------------------------------------------------
    #     # 5. (optional) extra global matcher – keep or remove
    #     # ---------------------------------------------------------
    #     # mmd_value = self.MMD.compute(x_tuple, attrs_rec)        # – REMOVED
    #     mmd_value = 0.0  # ★ set zero or keep if still wanted

    #     # ---------------------------------------------------------
    #     # 6. KL divergence (unchanged)
    #     # ---------------------------------------------------------
    #     kl_loss = self._kl_divergence_loss(mean, logvar)

    #     # ---------------------------------------------------------
    #     # 7. total loss =  joint NLL  +  β·KL  +  attribute BCE  +  (opt) MMD
    #     # ---------------------------------------------------------
    #     loss = rec_loss + kl_w * kl_loss + attr_loss + mmd_value  # ★ NEW formula

    #     # ---------------------------------------------------------
    #     # 8. logging
    #     # ---------------------------------------------------------
    #     self.log_dict(
    #         {
    #             f"{stage}/loss": loss,
    #             f"{stage}/nll_joint": rec_loss,  # ★ renamed metric
    #             f"{stage}/attr_loss": attr_loss,
    #             f"{stage}/kl_loss": kl_loss,
    #             "kl_weight": kl_w,
    #         },
    #         on_epoch=True,
    #         prog_bar=(stage == "train"),
    #     )
    #     # attribute-level breakdown (unchanged)
    #     # self.log_dict(
    #     #     {
    #     #         f"{stage}/rec/cat_attrs_loss": cat_attrs["cat_attrs"],
    #     #         f"{stage}/rec/num_attrs_loss": cat_attrs["num_attrs"],
    #     #     },
    #     #     on_epoch=True,
    #     # )
    #     return loss

    def kl_divergence_loss_fn(self, mean, var):
        var = torch.clamp(var, min=1e-9)
        return -torch.sum(1 + torch.log(var**2) - mean**2 - var**2)


    def reconstruction_loss_fn(self, x_rec, x):
        train_dataset = self.trainer.datamodule.data_train
        cat_attrs_loss, num_attrs_loss, cf_loss, ts_loss, res_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        cat_attr_loss_fn = nn.BCELoss(reduction="sum")
        num_attr_loss_fn = nn.MSELoss(reduction="sum")
        cf_loss_fn = nn.BCELoss(reduction="sum")
        ts_loss_fn = nn.MSELoss(reduction="sum")
        res_loss_fn = nn.BCELoss(reduction="sum")

        attrs_rec, acts_rec, ts_rec, ress_rec = x_rec
        attrs, acts, ts, ress = x

        # convert categorical attrs to one-hot encoding
        for attr_name, attr_val in attrs.items():
            if attr_name in train_dataset.s2i:  # it is a categorical attr
                attrs[attr_name] = (
                    F.one_hot(
                        attr_val.to(torch.int64),
                        num_classes=len(train_dataset.s2i[attr_name]),
                    )
                    .to(torch.float32)
                    .to(self.device)
                )

        # turn all PAD activities into EOT activities
        acts = torch.where(
            acts == train_dataset.activity2n[train_dataset.PADDING_ACTIVITY],
            train_dataset.activity2n[train_dataset.EOT_ACTIVITY],
            acts,
        ).to(self.device)
        # convert acts to one-hot encoding
        acts = F.one_hot(
            acts.to(torch.int64), num_classes=self.dataset_info.num_activities
        ).to(torch.float32)

        # attrs
        for attr_name, attr_val in attrs.items():
            if attr_name in train_dataset.s2i:
                cat_attrs_loss += cat_attr_loss_fn(attrs_rec[attr_name], attr_val)
            else:
                attrs_rec[attr_name] = attrs_rec[attr_name].squeeze()
                num_attrs_loss += num_attr_loss_fn(attrs_rec[attr_name], attr_val).to(
                    torch.float32
                )

        # acts
        cf_loss += cf_loss_fn(acts_rec, acts)

        # ts
        lens = acts.argmax(dim=2).argmax(dim=1)
        # use lens and compute loss only for the first lens elements
        for i in range(len(lens)):
            if lens[i] < ts_rec.size(1):
                ts_rec[i, lens[i] :] = 0.0

        ts_loss += ts_loss_fn(ts_rec, ts)

        # ress
        ress = torch.where(
            ress == train_dataset.resource2n[train_dataset.PADDING_RESOURCE],
            train_dataset.resource2n[train_dataset.EOT_RESOURCE],
            ress,
        ).to(self.device)
        ress = F.one_hot(
            ress.to(torch.int64), num_classes=self.dataset_info.num_resources
        ).to(torch.float32)

        res_loss += res_loss_fn(ress_rec, ress)

        # sum up loss components
        loss = cat_attrs_loss + num_attrs_loss + cf_loss + ts_loss + res_loss

        return loss, torch.tensor(
            [cat_attrs_loss, num_attrs_loss, cf_loss, ts_loss, res_loss]
        )


    def on_train_epoch_start(self):
        warmup = self.hparams.flow_warmup_epochs
        if warmup is not None and warmup > 0:
            if self.current_epoch < warmup:  # ─ Stage 1
                for n, p in self.model.decoder.named_parameters():
                    p.requires_grad = False if "flow." in n else True
                for p in self.model.encoder.parameters():
                    p.requires_grad = True
            else:  # ─ Stage 2
                for p in self.model.encoder.parameters():
                    p.requires_grad = False
                for n, p in self.model.decoder.named_parameters():
                    p.requires_grad = True if "flow." in n else False

    def flow_shared_step(self, batch, stage: str, kl_w: float) -> torch.Tensor:
        """
        Shared step for flow-based CVAE that properly handles the flow log probability

        Args:
            batch: Input batch with activities, timestamps, resources
            stage: 'train', 'val', or 'test'
            kl_w: Weight for KL divergence term

        Returns:
            Total loss value
        """

        x, c = batch

        mean, logvar = self.model.encode(x, c)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon  # reparametrization trick
        tuple_inputs = x[1:]  # (acts, dts, ress)

        if stage == "train":
            tf_ratio = self.get_teacher_forcing_ratio(self.current_epoch)
            attrs_rec, acts_rec, ts_rec, ress_rec, log_px = self.model.decode(
                z, c, tuple_inputs, tf_ratio=tf_ratio
            )

        if stage in ["val", "test"]:
            tf_ratio = self.get_teacher_forcing_ratio(self.current_epoch)

            attrs_rec, acts_rec, ts_rec, ress_rec, log_px = self.model.decode(
                z, c, tuple_inputs, tf_ratio=tf_ratio
            )

        # Log basic encoder statistics
        self.log_dict(
            {
                f"{stage}/mean": mean.mean(),
                f"{stage}/logvar": logvar.mean(),
            },
            prog_bar=(stage == "train"),
        )

        # KL divergence loss (unchanged)
        kl_loss, dimwise_kl = self.kl_divergence_loss_fn_flow_2(
            mean, logvar, free_bits=self.hparams.free_bits
        )

        # Flow log probability is already computed - higher is better, so we negate it
        # The log_px comes directly from the flow decoder
        flow_nll = -log_px.mean()  # Negative log likelihood

        # Optional: Calculate additional reconstruction metrics for monitoring
        # These don't affect training but are useful for comparison with baseline
        
            # Convert logits to probabilities for reconstruction metrics
        # acts_rec_probs = F.softmax(acts_rec, dim=-1)
        # ress_rec_probs = F.softmax(ress_rec, dim=-1)

            
        # ts_rec = ts_rec.squeeze(-1)
        # x_rec_for_metrics = (attrs_rec, acts_rec_probs, ts_rec, ress_rec_probs)
        # rec_metrics, rec_components = self.reconstruction_loss_fn(x_rec_for_metrics, x)

   
        rec_loss, rec_components = self.reconstruction_loss_flow_fn(
            (attrs_rec, acts_rec, ts_rec, ress_rec), x
        )
       
        # Total loss = flow_nll + KL + activity reconstruction (weighted if desired)

        loss = (
            flow_nll
            + kl_w * kl_loss
            + self.hparams.cat_loss_weight * rec_loss
    
        )

        # Log all metrics
        
        if rec_components is not None:
            self.log_dict(
                {   
                    f"{stage}/rec/cat_attrs_loss": rec_components['cat_attrs'],
                    f"{stage}/rec/num_attrs_loss": rec_components['num_attrs'],
                    f"{stage}/rec/cf_loss": rec_components['act'],
                    f"{stage}/rec/ts_loss": rec_components['ts'],
                    f"{stage}/rec/res_loss": rec_components['res'],
                },
                prog_bar=True,
            )

    
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/recon_loss": rec_loss,
                f"{stage}/flow_nll": flow_nll,
                f"{stage}/kl_loss": kl_loss,
                f"{stage}/kl_weight": kl_w,
            },
            prog_bar=True,
        )

        if stage == "val":
            dimwise_kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
            for i in range(min(5, mean.size(1))):
                self.log(f"latent/latent_dim_{i}_kl", dimwise_kl[:, i].mean())

            active_dims = (
                (dimwise_kl > 0.01).float().mean(dim=0)
            )  # Fraction of batch using each dim

            self.log_dict(
                {
                    f"latent/active_dims_mean": active_dims.mean(),
                    f"latent/posterior_collapse_ratio": (active_dims < 0.1).float().mean(),
                }
            )
            
            # 1. KL divergence distribution
            kl_std = dimwise_kl.mean(dim=0).std()  # Std dev across dimensions
            self.log("latent/latent_kl_std", kl_std)
            
            # 3. Dimension activity heatmap (every N epochs)
            if self.current_epoch % 10 == 0:
                dim_activity = (dimwise_kl > 0.01).float().mean(dim=0)
                for i, activity in enumerate(dim_activity):
                    self.log(f"latent/dim_{i}_active_ratio", activity)
                    
            # 4. Track highly active dimensions (KL > 0.1)
            highly_active = ((dimwise_kl > 0.1).float().mean(dim=0)).mean()
            self.log("latent/highly_active_dims_ratio", highly_active)


        
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    
    def flow_shared_step2(self, batch, stage: str, kl_w: float) -> torch.Tensor:
        """
        Alternative shared step for flow-based CVAE that uses the exact same loss
        calculations as the standard VAE for direct comparison.
        
        Args:
            batch: Input batch with activities, timestamps, resources
            stage: 'train', 'val', or 'test'
            kl_w: Weight for KL divergence term
            
        Returns:
            Total loss value
        """
        # Unpack batch (same as standard VAE)
        x, c = batch
        
        # ===== FLOW-SPECIFIC ENCODING =====
        # Keep the flow model's encoding process
        mean, var = self.model.encode(x, c)  # Note: var is actually logvar in the encoder
        
        # Log encoder statistics (same as in both steps)
        self.log_dict(
            {
                f"{stage}/mean": mean.mean(),
                f"{stage}/var": var.mean(),  # Keeping original naming for consistency
            },
            on_epoch=True,
        )
        
        # ===== REPARAMETERIZATION =====
        # Use the SAME reparameterization as the original VAE (which is incorrect)
        # This is intentional to match the behavior exactly
        std = torch.exp(0.5 * var)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon  # reparametrization trick
        # epsilon = torch.randn_like(var)
        # z = mean + var*epsilon  # Using var directly like the original
        
        # ===== FLOW-SPECIFIC DECODING =====
        # Keep flow decoder but ignore the log_px
        if stage == "train":
            tf_ratio = self.get_teacher_forcing_ratio(self.current_epoch)
            attrs_rec, acts_rec, ts_rec, ress_rec, log_px = self.model.decode(
                z, c, x[1:], tf_ratio=tf_ratio
            )
        else:
            tf_ratio = self.get_teacher_forcing_ratio(self.current_epoch)
            attrs_rec, acts_rec, ts_rec, ress_rec, log_px = self.model.decode(
                z, c, x[1:], tf_ratio=tf_ratio
            )
            
        ts_rec = ts_rec.squeeze(-1) if ts_rec.dim() > 2 else ts_rec

        
        # ===== STANDARD VAE LOSSES =====
        # Use EXACTLY the same reconstruction loss function as the standard VAE
        #x_rec = (attrs_rec, acts_rec, ts_rec, ress_rec)
        x_rec = (attrs_rec, acts_rec, ts_rec, ress_rec)
        
        
        rec_loss, cat_attrs = self._reconstruction_loss(x_rec, x)
        
        # Use EXACTLY the same KL divergence calculation as the standard VAE
        kl_loss = self.kl_divergence_loss_fn(mean, var)
        
        # Combine losses the same way as the standard VAE
        flow_nll = -log_px.mean()
        normalized_rec_loss = rec_loss / 128
        loss =  kl_w * kl_loss + flow_nll + 0.1 * normalized_rec_loss

        # ===== LOGGING =====
        # Log the same metrics as the standard VAE
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/rec_loss": rec_loss,
                f"{stage}/normalized_rec_loss": normalized_rec_loss,
                f"{stage}/kl_loss": kl_loss,
                f"{stage}/kl_weight": kl_w,
                # Additional flow-specific metric for information
                f"{stage}/flow_nll": flow_nll,  # Just for reference
            },
            on_epoch=True,
            prog_bar=True,
        )
        
        # Log component losses the same way as the standard VAE
        self.log_dict(
            {
                f"{stage}/rec/cat_attrs_loss": cat_attrs["cat_attrs"],
                f"{stage}/rec/num_attrs_loss": cat_attrs["num_attrs"],
                f"{stage}/rec/cf_loss": cat_attrs["cf"],
                f"{stage}/rec/ts_loss": cat_attrs["ts"],
                f"{stage}/rec/res_loss": cat_attrs["res"],
            },
            on_epoch=True,
        )
        
        # Log learning rate
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def get_teacher_forcing_ratio(self, epoch: int) -> float:
        cfg = self.hparams.get("teacher_forcing", {})
        start = cfg.get("start_ratio", 1.0)
        end = cfg.get("end_ratio", 0.0)
        decay = cfg.get("decay_epochs", 30)
        hold = cfg.get("hold_epochs", 0)

        if epoch < hold:
            self.log("teacher_forcing_ratio", start, prog_bar=True)
            return start
        else:
            # Calculate relative epoch after hold period
            rel_epoch = epoch - hold
            
            # Cosine annealing schedule (smoother than linear)
            progress = min(rel_epoch / decay, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            # Fix the formula here - end + distance * decay factor
            ratio = end + (start - end) * cosine_decay
            
            self.log("teacher_forcing_ratio", ratio, prog_bar=True)
            return ratio


    def reconstruction_loss_flow_fn(self, x_rec, x):
        """Compute reconstruction losses optimized for flow-based models.
        
        Args:
            x_rec: Tuple of (attrs_rec, acts_rec, ts_rec, ress_rec) from model output
            x: Tuple of (attrs, acts, ts, ress) ground truth
            
        Returns:
            total_loss: Combined reconstruction loss
            components: Dictionary with individual loss components
        """
        attrs_rec, acts_rec, ts_rec, ress_rec = x_rec
        attrs, acts_gt, ts_gt, ress_gt = x
        
        dm = self.trainer.datamodule.data_train
        
        # Initialize component losses
        losses = {
            "cat_attrs": torch.tensor(0.0, device=self.device),
            "num_attrs": torch.tensor(0.0, device=self.device),
            "act": torch.tensor(0.0, device=self.device),
            "ts": torch.tensor(0.0, device=self.device),
            "res": torch.tensor(0.0, device=self.device)
        }
        
        # ===== 1. Categorical and Numerical Attributes =====
        if attrs and attrs_rec:  # Only process if attributes exist
            for name, val in attrs.items():
                if name in dm.s2i:  # Categorical attribute
                    # Use BCEWithLogitsLoss for categorical attributes (expects logits)
                    target = F.one_hot(
                        val.to(torch.int64),
                        num_classes=len(dm.s2i[name]),
                    ).float()
                    attr_loss = F.binary_cross_entropy_with_logits(
                        attrs_rec[name], target, reduction="mean"
                    )
                    losses["cat_attrs"] += attr_loss
                else:  # Numerical attribute
                    pred = attrs_rec[name].squeeze()
                    target = val.float().squeeze()
                    # MSE loss for numerical attributes
                    attr_loss = F.mse_loss(pred, target, reduction="mean")
                    losses["num_attrs"] += attr_loss
        
        # ===== 2. Activities (Cross Entropy) =====
        # Get padding and EOS indices
        pad_act = dm.activity2n.get(dm.PADDING_ACTIVITY, 0)
        
        # Compute cross-entropy loss for activities
        B, L, V_act = acts_rec.shape
        losses["act"] = F.cross_entropy(
            acts_rec.reshape(-1, V_act),  # [B*L, V_act]
            acts_gt.reshape(-1).long(),   # [B*L]
            ignore_index=pad_act,         # Ignore padding tokens
            reduction="mean"
        )
        
        # ===== 3. Timestamps (MSE with masking) =====
        # Create mask for valid positions (not padding)
        ts_mask = (acts_gt != pad_act).float()
        ts_rec = ts_rec.squeeze(-1)  # Ensure timestamps are 2D
        # Apply mask to timestamps loss
        ts_squared_diff = (ts_rec - ts_gt) ** 2
        masked_ts_loss = (ts_squared_diff * ts_mask).sum()
        valid_ts_positions = ts_mask.sum().clamp(min=1.0)  # Avoid division by zero
        losses["ts"] = masked_ts_loss / valid_ts_positions
        
        # ===== 4. Resources (Cross Entropy) =====
        # Get padding index for resources
        pad_res = dm.resource2n.get(dm.PADDING_RESOURCE, 0)
        
        # Compute cross-entropy loss for resources
        B, L, V_res = ress_rec.shape
        losses["res"] = F.cross_entropy(
            ress_rec.reshape(-1, V_res),  # [B*L, V_res]
            ress_gt.reshape(-1).long(),   # [B*L]
            ignore_index=pad_res,         # Ignore padding tokens
            reduction="mean"
        )
        
        # ===== 5. Combine losses =====
        # You can adjust weights here if needed
        total_loss = (
            losses["cat_attrs"] + 
            losses["num_attrs"] + 
            losses["act"] + 
            losses["ts"] + 
            losses["res"]
        )
        
        return total_loss, losses


    def flow_shared_step_simplex(self, batch, stage: str, kl_w: float) -> torch.Tensor:
        """
        Shared step for the Transformer–Simplex Flow CVAE (no scheduled sampling).
        Trains with pure teacher forcing and uses the joint flow likelihood bound.

        Args:
            batch: tuple (x, c) where x packs inputs and c is the conditional vector
                Expected x layout: x = (x0, acts, dts, ress, ...) so that x[1:] = (acts, dts, ress)
            stage: 'train', 'val', or 'test'
            kl_w:  KL weight (e.g., cyclical annealing factor)

        Returns:
            Total loss (torch.Tensor)
        """
        x, c = batch

        # ----- Encode & reparameterize -----
        mean, logvar = self.model.encode(x, c)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + std * eps

        # Ground-truth tuples for teacher-forced decoding (acts, dts, ress)
        tuple_inputs = x[1:]  # (acts_gt, dts_gt, ress_gt)

        # ----- Decode (pure teacher forcing; no tf_ratio) -----
        attrs_rec, acts_rec, ts_rec, ress_rec, log_px = self.model.decode(z, c, tuple_inputs)

        # ----- Basic encoder stats -----
        self.log_dict(
            {
                f"{stage}/mean": mean.mean(),
                f"{stage}/logvar": logvar.mean(),
            },
            prog_bar=(stage == "train"),
        )

        # ----- KL divergence (unchanged) -----
        kl_loss, dimwise_kl = self.kl_divergence_loss_fn_flow_2(
            mean, logvar, free_bits=self.hparams.free_bits
        )

        # ----- Flow negative log-likelihood (joint over a,r,t; already summed over time) -----
        flow_nll = -log_px.mean()  # minimize NLL

        # ----- Optional reconstruction loss (uses your existing routine) -----
        # NOTE: acts_rec/ress_rec are logits (log-probs). ts_rec is (B,T,1).
        # rec_loss, rec_components = self.reconstruction_loss_flow_fn(
        #     (attrs_rec, acts_rec, ts_rec, ress_rec), x
        # )

        # ----- Total loss -----
        loss = flow_nll + kl_w * kl_loss #+ self.hparams.cat_loss_weight * rec_loss

        # # ----- Log reconstruction components (if provided) -----
        # if rec_components is not None:
        #     self.log_dict(
        #         {
        #             f"{stage}/rec/cat_attrs_loss": rec_components.get("cat_attrs", torch.tensor(0.0, device=loss.device)),
        #             f"{stage}/rec/num_attrs_loss": rec_components.get("num_attrs", torch.tensor(0.0, device=loss.device)),
        #             f"{stage}/rec/cf_loss":        rec_components.get("act", torch.tensor(0.0, device=loss.device)),
        #             f"{stage}/rec/ts_loss":        rec_components.get("ts",  torch.tensor(0.0, device=loss.device)),
        #             f"{stage}/rec/res_loss":       rec_components.get("res", torch.tensor(0.0, device=loss.device)),
        #         },
        #         prog_bar=True,
        #     )

        # ----- Core logs -----
        self.log_dict(
            {
                f"{stage}/loss":       loss,
                # f"{stage}/recon_loss": rec_loss,
                f"{stage}/flow_nll":   flow_nll,
                f"{stage}/kl_loss":    kl_loss,
                f"{stage}/kl_weight":  kl_w,
            },
            prog_bar=True,
        )

        # ----- Extra diagnostics on validation -----
        if stage == "val":
            # Per-dimension KL (mean over batch)
            dimwise_kl_vals = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
            for i in range(min(5, mean.size(1))):
                self.log(f"latent/latent_dim_{i}_kl", dimwise_kl_vals[:, i].mean())

            active_dims = (dimwise_kl_vals > 0.01).float().mean(dim=0)
            self.log_dict(
                {
                    "latent/active_dims_mean": active_dims.mean(),
                    "latent/posterior_collapse_ratio": (active_dims < 0.1).float().mean(),
                }
            )

            kl_std = dimwise_kl_vals.mean(dim=0).std()
            self.log("latent/latent_kl_std", kl_std)

            if self.current_epoch % 10 == 0:
                dim_activity = (dimwise_kl_vals > 0.01).float().mean(dim=0)
                for i, activity in enumerate(dim_activity):
                    self.log(f"latent/dim_{i}_active_ratio", activity)

            highly_active = ((dimwise_kl_vals > 0.1).float().mean(dim=0)).mean()
            self.log("latent/highly_active_dims_ratio", highly_active)

        # ----- LR monitor -----
        opt = self.optimizers()
        if opt is not None and len(opt.param_groups) > 0:
            lr = opt.param_groups[0]["lr"]
            self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss


