"""Phase 2 M3 model: shared multitask MLP with explicit ablation controls.

Input is the ordered 520-value linear-positive RSSI vector (or 1,040 values
when an experiment enables missingness flags). Coordinates returned by the
model are standardized targets; callers must inverse-transform before scoring.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

import torch
from torch import nn


class _Trunk(nn.Sequential):
    """Configurable fully connected trunk with a DAE-compatible encoder view."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float,
                 batch_norm: bool, *, dropout_last: bool = False):
        layers: list[nn.Module] = []
        previous = input_dim
        for index, width in enumerate(hidden_dims):
            width = int(width)
            layers.append(nn.Linear(previous, width))
            regularize = dropout_last or index < len(hidden_dims) - 1
            if batch_norm and regularize:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if dropout > 0 and regularize:
                layers.append(nn.Dropout(float(dropout)))
            previous = width
        super().__init__(*layers)
        self.output_dim = previous


class MultitaskMLP(nn.Module):
    """Shared feature trunk and task-specific building, floor, and coordinate heads."""
    def __init__(self, input_dim: int = 520, hidden_dims: Sequence[int] = (512, 256, 128),
                 dropout: float = 0.3, batch_norm: bool = True,
                 floor_mode: str = "flat", loss_weighting: str = "fixed",
                 dropout_last: bool = False):
        super().__init__()
        if input_dim not in (520, 1040):
            raise ValueError("input_dim must be 520 or 1040")
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")
        if floor_mode not in {"flat", "hierarchical"}:
            raise ValueError("floor_mode must be 'flat' or 'hierarchical'")
        if loss_weighting not in {"fixed", "uncertainty"}:
            raise ValueError("loss_weighting must be 'fixed' or 'uncertainty'")
        self.input_dim = input_dim
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.dropout = float(dropout)
        self.batch_norm = bool(batch_norm)
        self.dropout_last = bool(dropout_last)
        self.embedding_dim = self.hidden_dims[-1]
        self.floor_mode = floor_mode
        self.loss_weighting = loss_weighting
        # Keep numeric Sequential state keys so M2 can inspect/copy encoder weights.
        self.trunk = _Trunk(input_dim, self.hidden_dims, dropout, batch_norm,
                            dropout_last=dropout_last)
        self.building_head = nn.Linear(self.embedding_dim, 3)
        if floor_mode == "flat":
            self.floor_head = nn.Linear(self.embedding_dim, 5)
            self.conditional_floor_heads = None
        else:
            self.floor_head = None
            self.conditional_floor_heads = nn.ModuleList(
                [nn.Linear(self.embedding_dim, 5) for _ in range(3)]
            )
        self.coordinate_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 64), nn.ReLU(), nn.Linear(64, 2)
        )
        self.log_task_vars = (
            nn.Parameter(torch.zeros(3)) if loss_weighting == "uncertainty" else None
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Return the shared embedding for M2's downstream localization models."""
        return self.trunk(features)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(features)
        building_logits = self.building_head(latent)
        if self.floor_mode == "flat":
            assert self.floor_head is not None
            floor_logits = self.floor_head(latent)
        else:
            assert self.conditional_floor_heads is not None
            conditional_logits = torch.stack(
                [head(latent) for head in self.conditional_floor_heads], dim=1
            )
            # Soft predicted-building routing avoids oracle labels at validation/inference.
            building_probability = torch.softmax(building_logits, dim=1)
            floor_logits = (conditional_logits * building_probability.unsqueeze(-1)).sum(dim=1)
        return {
            "building_logits": building_logits,
            "floor_logits": floor_logits,
            "coordinates": self.coordinate_head(latent),
        }


def make_model(input_dim: int = 520, dropout: float = 0.3, batch_norm: bool = True,
               seed: int = 42, hidden_dims: Sequence[int] | None = None,
               width: int | None = None, depth: int | None = None, floor_mode: str = "flat",
               loss_weighting: str = "fixed", dropout_last: bool = False,
               independent_tasks: bool = False, **_: object) -> nn.Module:
    """Seeded factory used by M1/M2/M4 and the M3 experiment runner.

    ``hidden_dims`` takes precedence. Otherwise ``depth`` and ``width`` expose
    A9's 2/3/4 layer and 256/512/1024 width settings.
    """
    if hidden_dims is None:
        if width is None and depth is None:
            hidden_dims = (512, 256, 128)
        else:
            width = 512 if width is None else width
            depth = 3 if depth is None else depth
            if depth not in (2, 3, 4):
                raise ValueError("depth must be 2, 3, or 4")
            if width not in (256, 512, 1024):
                raise ValueError("width must be 256, 512, or 1024")
            hidden_dims = tuple([int(width)] * depth)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = MultitaskMLP(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout,
            batch_norm=batch_norm, floor_mode=floor_mode,
            loss_weighting=loss_weighting, dropout_last=dropout_last,
        )
        if independent_tasks:
            model = IndependentTaskMLP.from_shared(model)
    return model


class IndependentTaskMLP(nn.Module):
    """A1 control: three separate copies of a shared model's task-sized trunk."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float,
                 batch_norm: bool, dropout_last: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(int(w) for w in hidden_dims)
        self.building_trunk = _Trunk(input_dim, hidden_dims, dropout, batch_norm, dropout_last=dropout_last)
        self.floor_trunk = _Trunk(input_dim, hidden_dims, dropout, batch_norm, dropout_last=dropout_last)
        self.coordinate_trunk = _Trunk(input_dim, hidden_dims, dropout, batch_norm, dropout_last=dropout_last)
        width = self.hidden_dims[-1]
        self.building_head = nn.Linear(width, 3)
        self.floor_head = nn.Linear(width, 5)
        self.coordinate_head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 2))
        self.log_task_vars = None

    @classmethod
    def from_shared(cls, shared: MultitaskMLP) -> "IndependentTaskMLP":
        # Derive matching modules without consuming or changing global RNG state.
        model = cls(shared.input_dim, shared.hidden_dims, shared.dropout,
                    shared.batch_norm, dropout_last=shared.dropout_last)
        model.building_head = shared.building_head
        model.floor_head = shared.floor_head if shared.floor_head is not None else nn.Linear(shared.embedding_dim, 5)
        model.coordinate_head = shared.coordinate_head
        return model

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.coordinate_trunk(features)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "building_logits": self.building_head(self.building_trunk(features)),
            "floor_logits": self.floor_head(self.floor_trunk(features)),
            "coordinates": self.coordinate_head(self.encode(features)),
        }


def multitask_loss(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
                   model: nn.Module | None = None,
                   weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equal weighted CE + CE + coordinate Huber, or learned uncertainty weights."""
    building = nn.functional.cross_entropy(outputs["building_logits"], batch["building"])
    floor = nn.functional.cross_entropy(outputs["floor_logits"], batch["floor"])
    coords = nn.functional.huber_loss(outputs["coordinates"], batch["coordinates"])
    components = {"building": building, "floor": floor, "coordinates": coords}
    log_vars = getattr(model, "log_task_vars", None) if model is not None else None
    if log_vars is not None:
        total = torch.sum(torch.exp(-log_vars) * torch.stack((building, floor, coords)) + log_vars)
    else:
        total = weights[0] * building + weights[1] * floor + weights[2] * coords
    return total, components


build_multitask_model = make_model
