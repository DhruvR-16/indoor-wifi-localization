"""TEMPORARY STUB FOR M3 MULTITASK MODEL - MUST BE REPLACED BY M3'S REAL FACTORY.

This module provides a strictly compliant, lightweight PyTorch multitask MLP stub
implementing the agreed M3 interface so that M1 data, regularization, and augmentation
sweeps can be smoke tested, verified, and benchmarked without blocking on M3's final
implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultitaskMLPStub(nn.Module):
    """Temporary multitask MLP stub adhering to the M3 interface contract."""

    def __init__(
        self,
        input_dim: int = 520,
        dropout: float = 0.2,
        batch_norm: bool = True,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        trunk_out_dim = hidden_dims[-1]

        # Multi-task heads:
        # Building: 3 classes (0, 1, 2)
        self.building_head = nn.Linear(trunk_out_dim, 3)
        # Floor: 5 classes (0, 1, 2, 3, 4)
        self.floor_head = nn.Linear(trunk_out_dim, 5)
        # Coordinates: 2 continuous targets (normalized LONGITUDE, LATITUDE)
        self.coordinate_head = nn.Linear(trunk_out_dim, 2)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Return latent representation from the shared trunk (for M2 hybrid kNN)."""
        return self.trunk(features)

    def forward(
        self, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute logits and coordinates."""
        latent = self.encode(features)
        building_logits = self.building_head(latent)
        floor_logits = self.floor_head(latent)
        coordinates = self.coordinate_head(latent)
        return {
            "building_logits": building_logits,
            "floor_logits": floor_logits,
            "coordinates": coordinates,
        }


def build_multitask_model(
    input_dim: int = 520,
    dropout: float = 0.2,
    batch_norm: bool = True,
    seed: int = 42,
    hidden_dims: tuple[int, ...] = (256, 128),
    **kwargs,
) -> MultitaskMLPStub:
    """Factory creating the multitask model.
    
    WARNING: THIS IS A TEMPORARY STUB. REPLACE WITH M3'S FACTORY ONCE DELIVERED.
    """
    torch.manual_seed(seed)
    return MultitaskMLPStub(
        input_dim=input_dim,
        dropout=dropout,
        batch_norm=batch_norm,
        hidden_dims=hidden_dims,
    )

