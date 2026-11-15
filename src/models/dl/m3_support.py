"""Small M3-only evaluation/device helpers shared by the M3 OOF runner."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import pandas as pd
import torch
from torch import nn


def select_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prediction_frame(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
                     coordinate_mean: np.ndarray, coordinate_scale: np.ndarray) -> pd.DataFrame:
    building = outputs["building_logits"].argmax(dim=1).detach().cpu().numpy()
    floor = outputs["floor_logits"].argmax(dim=1).detach().cpu().numpy()
    coords = outputs["coordinates"].detach().cpu().numpy()
    truth_coords = batch["coordinates"].detach().cpu().numpy()
    coords = coords * coordinate_scale + coordinate_mean
    truth_coords = truth_coords * coordinate_scale + coordinate_mean
    return pd.DataFrame({
        "BUILDINGID": batch["building"].detach().cpu().numpy(),
        "BUILDINGID_PRED": building,
        "FLOOR": batch["floor"].detach().cpu().numpy(),
        "FLOOR_PRED": floor,
        "LONGITUDE_TRUE": truth_coords[:, 0],
        "LATITUDE_TRUE": truth_coords[:, 1],
        "LONGITUDE_PRED": coords[:, 0],
        "LATITUDE_PRED": coords[:, 1],
    })
