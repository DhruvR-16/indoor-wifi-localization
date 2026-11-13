"""RSSI input representations for Phase 2 ablation A4.

With ``P(x) = clip(x, -110, 0) + 110`` the positive reading of a detected AP
and ``R = 110``:

* ``linear``       -> ``P(x) / R``                       (the M1 contract view)
* ``exponential``  -> ``exp(P(x) / alpha) / exp(R / alpha)``, alpha = 24
* ``powed``        -> ``(P(x) / R) ** beta``,                 beta = e

Definitions follow Torres-Sospedra et al. (2015). Every view maps the
not-detected sentinel to exactly 0.0 and detected readings into (0.0, 1.0],
and is a pure per-reading function, so nothing is fitted and nothing can
cross a fold boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.data.load import MISSING_RSSI
from src.data.phase2 import CEIL_RSSI, FLOOR_RSSI

REPRESENTATIONS = ("linear", "exponential", "powed")

#: Decay constant of the exponential view, in dB (Torres-Sospedra et al., 2015).
DEFAULT_ALPHA = 24.0
#: Exponent of the powed view (Torres-Sospedra et al., 2015).
DEFAULT_BETA = math.e


@dataclass(frozen=True)
class RepresentationSpec:
    """A4 arm: one RSSI view, optionally concatenated with missingness flags."""

    name: str = "linear"
    include_missing_mask: bool = False
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    floor_rssi: float = FLOOR_RSSI
    ceil_rssi: float = CEIL_RSSI
    missing_sentinel: float = float(MISSING_RSSI)

    def __post_init__(self) -> None:
        if self.name not in REPRESENTATIONS:
            raise ValueError(
                f"Unknown representation {self.name!r}; expected one of {REPRESENTATIONS}"
            )
        if self.ceil_rssi <= self.floor_rssi:
            raise ValueError("ceil_rssi must exceed floor_rssi")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")

    @property
    def exp_id_suffix(self) -> str:
        return f"{self.name}{'_mask' if self.include_missing_mask else ''}"

    def input_dim(self, wap_count: int = 520) -> int:
        return wap_count * (2 if self.include_missing_mask else 1)

    def __call__(self, raw_dbm: np.ndarray) -> np.ndarray:
        return transform_rssi(raw_dbm, self)


def transform_rssi(raw_dbm: np.ndarray, spec: RepresentationSpec) -> np.ndarray:
    """Apply ``spec`` to raw dBm readings, returning float32 in ``[0, 1]``.

    Works on a single scan ``(D,)`` or a batch ``(N, D)``. When
    ``spec.include_missing_mask`` is set, the detection flags are concatenated
    along the last axis, yielding width ``2 * D``.
    """
    raw = np.asarray(raw_dbm, dtype=np.float32)
    observed = raw != spec.missing_sentinel
    out = np.zeros_like(raw, dtype=np.float32)
    span = spec.ceil_rssi - spec.floor_rssi

    if np.any(observed):
        positive = (
            np.clip(raw[observed], spec.floor_rssi, spec.ceil_rssi) - spec.floor_rssi
        )
        if spec.name == "linear":
            values = positive / span
        elif spec.name == "exponential":
            # Normalised by the value at full scale so the view lands in (0, 1].
            values = np.exp(positive / spec.alpha) / math.exp(span / spec.alpha)
        else:  # powed
            values = np.power(positive / span, spec.beta)
        out[observed] = values.astype(np.float32)

    if spec.include_missing_mask:
        return np.concatenate([out, observed.astype(np.float32)], axis=-1)
    return out
