import torch
from torch import nn
from nflows.transforms import (
    CompositeTransform,
    ReversePermutation,
    RandomPermutation,
    LULinear,
    ActNorm,
    PiecewiseRationalQuadraticCouplingTransform
)
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.nn.nets import ResidualNet


def build_rqs_flow(
    in_features: int,
    context_features: int,
    layers: int = 12,
    num_bins: int = 24,
    tail_bound: float = 8.0,
    cond_hidden: int = 256,
    cond_blocks: int = 2,
    cond_dropout: float = 0.1,
    use_alternating_mask: bool = True,
    use_random_perm_each_layer: bool = True,
):
    transforms = []

    mask_even = torch.zeros(in_features)
    mask_even[::2] = 1.0
    mask_odd = 1.0 - mask_even

    def make_net(in_, out_):
        return ResidualNet(
            in_features=in_,
            out_features=out_,
            hidden_features=cond_hidden,
            context_features=context_features,
            num_blocks=cond_blocks,
            activation=nn.SiLU(),
            dropout_probability=cond_dropout,
            use_batch_norm=False,
        )

    for k in range(layers):
        mask = mask_odd if (k % 2 == 1 and use_alternating_mask) else mask_even
        transforms += [
            ActNorm(in_features),
            LULinear(in_features),
            RandomPermutation(in_features) if use_random_perm_each_layer else ReversePermutation(in_features),
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_net,
                tails="linear",
                tail_bound=tail_bound,
                num_bins=num_bins,
                min_bin_width=1e-3,
                min_bin_height=1e-3,
                min_derivative=1e-3,
            ),
        ]

    return Flow(CompositeTransform(transforms), StandardNormal(shape=[in_features]))


def set_actnorm_identity(flow):
    for m in flow._transform.modules():
        if isinstance(m, ActNorm):
            # Mark initialized so it does not shift data on first batch
            m.initialized = torch.tensor(True)