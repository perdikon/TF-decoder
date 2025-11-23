import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


# =========================
# Learned orthogonal rotation via Householder stack
# =========================
class LearnedOrthogonal(nn.Module):
    """Product of k Householder reflections (orthogonal matrix)."""
    def __init__(self, d: int, k: int = 2):
        super().__init__()
        self.v = nn.Parameter(torch.randn(k, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d)
        for i in range(self.v.size(0)):
            v = F.normalize(self.v[i], dim=0)
            x = x - 2 * (x @ v)[:, None] * v[None, :]
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # reverse order
        for i in reversed(range(self.v.size(0))):
            v = F.normalize(self.v[i], dim=0)
            x = x - 2 * (x @ v)[:, None] * v[None, :]
        return x


# =========================
# ILR / ALR and Helmert basis
# =========================
def helmert_basis(K: int, device=None, dtype=None):
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    B = torch.zeros(K, K - 1, device=device, dtype=dtype)
    for j in range(1, K):
        v = torch.zeros(K, device=device, dtype=dtype)
        v[:j] = 1.0 / j
        v[j] = -1.0
        v = v * math.sqrt(j / (j + 1.0))
        B[:, j - 1] = v
    return B  # columns orthonormal

def ilr(x: torch.Tensor, B: torch.Tensor, eps=1e-8):
    x = x.clamp_min(eps)
    logx = torch.log(x)
    clr = logx - logx.mean(dim=-1, keepdim=True)
    return clr @ B

def inv_ilr(y: torch.Tensor, B: torch.Tensor):
    clr = y @ B.T
    logx = clr - clr.mean(dim=-1, keepdim=True)
    x = torch.exp(logx)
    return x / x.sum(dim=-1, keepdim=True)

def logdet_jac_ilr(x: torch.Tensor, eps=1e-8, add_const=False, B: torch.Tensor = None):
    val = -torch.sum(torch.log(x.clamp_min(eps)), dim=-1)
    # const omitted (doesn't affect differences)
    return val

# ---------- ALR ----------
def alr(x, eps=1e-8):
    x = x.clamp_min(eps)
    return torch.log(x[..., :-1]) - torch.log(x[..., -1:])

def inv_alr(y):
    z = torch.cat([y, torch.zeros_like(y[..., :1])], dim=-1)
    z = torch.exp(z)
    return z / z.sum(dim=-1, keepdim=True)

def logdet_jac_alr(x, eps=1e-8):
    return -torch.sum(torch.log(x.clamp_min(eps)), dim=-1)


# =========================
# Dirichlet dequantization (padding-aware)
# =========================
@torch.no_grad()
def _build_concentration(indices, K, pad_idx,
                         alpha_main, alpha_noise,
                         pad_policy="uniform",
                         alpha_pad_main=200.0,
                         alpha_pad_noise=1.0):
    B = indices.shape[0]
    conc = torch.full((B, K), alpha_noise, device=indices.device, dtype=torch.float)
    nonpad_mask = (indices != pad_idx)
    if nonpad_mask.any():
        conc[nonpad_mask] = alpha_noise
        conc[nonpad_mask, indices[nonpad_mask]] = alpha_main
    pad_mask = ~nonpad_mask
    if pad_mask.any():
        if pad_policy == "uniform":
            conc[pad_mask] = alpha_pad_noise
        elif pad_policy == "pad_peak":
            conc[pad_mask] = alpha_pad_noise
            conc[pad_mask, pad_idx] = alpha_pad_main
        else:
            raise ValueError(f"Unknown pad_policy={pad_policy}")
    conc.clamp_(min=1e-6)
    return conc

def dirichlet_dequantize_with_pad(indices, K, pad_idx,
                                  alpha_main=60.0, alpha_noise=0.5,
                                  pad_policy="pad_peak",
                                  alpha_pad_main=200.0, alpha_pad_noise=1.0):
    conc = _build_concentration(
        indices, K, pad_idx,
        alpha_main, alpha_noise,
        pad_policy=pad_policy,
        alpha_pad_main=alpha_pad_main,
        alpha_pad_noise=alpha_pad_noise
    )
    dist = Dirichlet(concentration=conc)
    ya = dist.rsample()
    log_q = dist.log_prob(ya)
    valid_mask = (indices != pad_idx)
    return ya, log_q, valid_mask, dist