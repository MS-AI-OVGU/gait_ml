# predict_gait_events.py
from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from typing import List, Optional, Sequence, Union, Tuple

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from gait_ml.model.litmodel import LitSeq2Seq
from gait_ml.data.dataset import GaitDataset
from gait_ml.data.datamodule import GaitDataModule
from gait_ml import evaluate


@dataclass
class PredictConfig:
    # Inputs
    files_glob: str = "../data/dataset2/10092025/imu/Mobilephone/Termin_1_Vicon/ID_01_T1/*_2mW_IPhone.xls"
    ckpt_path: str = "/path/to/model.ckpt"

    # Data params (kept close to your notebook)
    batch_size: int = 32
    window_size: int = 256
    step_size: int = 256
    expand_labels: int = 0
    acc_sheet_name: str = "Linear Accelerometer"
    gyr_sheet_name: str = "Gyroscope"
    num_workers: int = 1

    # Z-scale
    zscale: bool = True
    zscale_mean: Tuple[float] = (
        -0.26654707,
        -0.02442654,
        0.37051307,
        -0.00488071,
        1.36632781,
        0.86179046,
    )
    zscale_std: Tuple[float] = (
        1.42070944,
        2.89440634,
        1.53666348,
        5.25694552,
        3.22176259,
        5.12151813,
    )

    # Runtime
    device: str = "auto"  # "auto" | "cpu" | "cuda"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    return torch.device(device)


def predict_gait_event_labels(
    ckpt_path: str,
    files: Sequence[str],
    *,
    batch_size: int = 32,
    window_size: int = 256,
    step_size: int = 256,
    expand_labels: int = 0,
    acc_sheet_name: str = "Linear Accelerometer",
    gyr_sheet_name: str = "Gyroscope",
    num_workers: int = 1,
    zscale: bool = True,
    zscale_stats: Optional[dict] = None,
    device: Union[str, torch.device] = "auto",
) -> np.ndarray:
    """
    Given a trained model checkpoint and input files, predict gait event labels.
    """
    assert len(files) == 1, (
        "Currently only single-file prediction is supported due to signal merging."
    )
    if not files:
        raise ValueError("No input files provided.")

    if isinstance(device, str):
        device = _resolve_device(device)

    # Load Lightning model from checkpoint
    lit = LitSeq2Seq.load_from_checkpoint(ckpt_path).to(device)
    lit.eval()

    # Dataset / dataloader (keep your explicit test_dataset override)
    test_dm = GaitDataModule(
        list(files),
        batch_size=batch_size,
        window_size=window_size,
        step_size=step_size,
        train_idx=None,
        val_idx=None,
        test_idx=[0],
        expand_labels=expand_labels,
        acc_sheet_name=acc_sheet_name,
        num_workers=num_workers,
        zscale=zscale,
        zscale_stats=zscale_stats,
    )
    test_ds = GaitDataset(
        csv_files=list(files),
        window_size=window_size,
        step_size=step_size,
        expand_labels=expand_labels,
        acc_sheet_name=acc_sheet_name,
        gyr_sheet_name=gyr_sheet_name,
        zscale=zscale,
        zscale_stats=zscale_stats,
    )
    test_dm.test_dataset = test_ds
    loader = test_dm.test_dataloader()

    # Inference
    preds = []
    with torch.inference_mode():
        for batch in loader:
            x, x_len = batch[0].to(device), batch[1].to(device)
            logits = lit(x, x_len)  # same call you used in the notebook
            preds.append(logits.detach().cpu())

    pred = torch.cat(preds, dim=0).reshape(-1, 3).argmax(-1).numpy()
    merged = evaluate.merge_clustered_events(pred)
    return merged


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    # Allow running with either structured defaults or a YAML/CLI override.
    base = OmegaConf.structured(PredictConfig)
    cfg = OmegaConf.merge(base, cfg)

    files = sorted(glob(cfg.files_glob))
    zscale_stats = (
        {
            "mean": np.asarray(cfg.zscale_mean, dtype=np.float32),
            "std": np.asarray(cfg.zscale_std, dtype=np.float32),
        }
        if cfg.zscale
        else None
    )

    merged = predict_gait_event_labels(
        ckpt_path=cfg.ckpt_path,
        files=files,
        batch_size=cfg.batch_size,
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        expand_labels=cfg.expand_labels,
        acc_sheet_name=cfg.acc_sheet_name,
        gyr_sheet_name=cfg.gyr_sheet_name,
        num_workers=cfg.num_workers,
        zscale=cfg.zscale,
        zscale_stats=zscale_stats,
        device=cfg.device,
    )

    # Minimal CLI output (you can redirect to file)
    print("n_files:", len(files))
    print("n_preds:", int(merged.shape[0]))
    print("merged_preds_head:", merged[:50].tolist())


if __name__ == "__main__":
    main()
