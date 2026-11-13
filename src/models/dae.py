"""Denoising autoencoder for Phase 2 ablation A7.

The encoder is architecture-matched to the multitask trunk - both are
``520 -> 256 -> 128`` with the same per-layer module order - so the only
difference between A7's two arms is whether the trunk starts from random or
pretrained weights. `load_encoder_into_trunk` raises rather than skipping or
reshaping a layer to force a fit.

Observed and missing entries are scored separately and combined with a stated
weighting, because ~96% of a scan is not detected and would otherwise dominate
a single reconstruction number.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _mlp_block(
    in_dim: int, out_dim: int, dropout: float, batch_norm: bool
) -> list[nn.Module]:
    """One trunk layer, in the exact module order used by the multitask trunk."""
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(nn.ReLU())
    if dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))
    return layers


class DenoisingAutoencoder(nn.Module):
    """``520 -> 256 -> 128`` encoder with a mirrored decoder.

    The decoder ends in a sigmoid because every Member 2 representation
    (linear / exponential / powed) lands in ``[0, 1]``.
    """

    def __init__(
        self,
        input_dim: int = 520,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = self.hidden_dims[-1]

        encoder_layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in self.hidden_dims:
            encoder_layers += _mlp_block(in_dim, h_dim, dropout, batch_norm)
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        for h_dim in reversed(self.hidden_dims[:-1]):
            decoder_layers += _mlp_block(in_dim, h_dim, dropout, batch_norm)
            in_dim = h_dim
        decoder_layers += [nn.Linear(in_dim, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(features))


@dataclass(frozen=True)
class MaskedReconstruction:
    """Reconstruction loss split by whether the AP was detected in the target."""

    total: torch.Tensor
    observed_mse: torch.Tensor
    missing_mse: torch.Tensor
    observed_entries: int
    missing_entries: int


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
    observed_weight: float = 1.0,
    missing_weight: float = 0.1,
) -> MaskedReconstruction:
    """Weighted MSE over observed and missing entries, reported separately.

    A scan detects ~18 of 520 APs, so an unweighted MSE is ~96% driven by
    predicting zeros. ``missing_weight`` down-weights that bulk term while
    still asking the decoder to respect sparsity; both parts are returned so
    the split can be recorded per run.
    """
    squared_error = (reconstruction - target) ** 2
    observed_mask = observed_mask.bool()
    n_observed = int(observed_mask.sum().item())
    n_missing = int((~observed_mask).sum().item())

    zero = torch.zeros((), dtype=reconstruction.dtype, device=reconstruction.device)
    observed_mse = squared_error[observed_mask].mean() if n_observed else zero
    missing_mse = squared_error[~observed_mask].mean() if n_missing else zero
    total = observed_weight * observed_mse + missing_weight * missing_mse

    return MaskedReconstruction(
        total=total,
        observed_mse=observed_mse,
        missing_mse=missing_mse,
        observed_entries=n_observed,
        missing_entries=n_missing,
    )


def load_encoder_into_trunk(
    trunk: nn.Module,
    encoder_state: dict[str, torch.Tensor],
    strict: bool = True,
) -> dict[str, tuple[int, ...]]:
    """Copy DAE encoder weights into a multitask trunk, refusing silent mismatches.

    Returns the mapping of transferred parameter names to their shapes so the
    transfer can be recorded in the run metadata.
    """
    trunk_state = trunk.state_dict()
    missing = [k for k in trunk_state if k not in encoder_state]
    unexpected = [k for k in encoder_state if k not in trunk_state]
    mismatched = {
        k: (tuple(trunk_state[k].shape), tuple(encoder_state[k].shape))
        for k in trunk_state
        if k in encoder_state
        and tuple(trunk_state[k].shape) != tuple(encoder_state[k].shape)
    }

    if strict and (missing or unexpected or mismatched):
        raise ValueError(
            "DAE encoder is not architecture-matched to the trunk; refusing to "
            "skip or reshape layers. "
            f"missing={missing} unexpected={unexpected} mismatched={mismatched}"
        )

    transferred = {
        k: tuple(encoder_state[k].shape)
        for k in trunk_state
        if k in encoder_state
        and tuple(trunk_state[k].shape) == tuple(encoder_state[k].shape)
    }
    trunk.load_state_dict({**trunk_state, **{k: encoder_state[k] for k in transferred}})
    return transferred


def build_dae(
    input_dim: int = 520,
    hidden_dims: tuple[int, ...] = (256, 128),
    dropout: float = 0.2,
    batch_norm: bool = True,
    seed: int = 42,
) -> DenoisingAutoencoder:
    torch.manual_seed(seed)
    return DenoisingAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_norm=batch_norm,
    )
