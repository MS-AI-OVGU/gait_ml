"""
motoric_feature_pipeline.py

End-to-end feature extraction pipeline for motoric tests (IMU):

- 2mW gait (iPhone)             → gait_imu_parameters.get_gait_features
- TUG single-task (TUGst)       → tug_imu_parameters.compute_tug_imu_parameters
- Trunk motion ROM (all flex.)  → trunk_motion_imu.compute_all_trunk_motion_for_subject
- STS-5r                        → sts_imu_parameters.get_sts_imu_features_single

Design goals:
- Works for ONE SUBJECT or ALL SUBJECTS
- Robust: **no single test failure breaks the pipeline**
- Error tables collected separately
- All feature tables aligned on `id`
"""


from __future__ import annotations

import os
from functools import reduce
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Imports from your existing feature modules
# ---------------------------------------------------------------------
from src.gait_ml.features.tug_imu_parameters import compute_tug_imu_parameters
from src.gait_ml.features.trunk_motion_imu import compute_all_trunk_motion_for_subject
from src.gait_ml.features.sts_imu_parameters import get_sts_imu_features_single
from src.gait_ml.features.gait_imu_parameters import (
    get_gait_features, compute_gait_imu_parameters
)
from configs.config import DATASETS_02_T2


# ---------------------------------------------------------------------
# File patterns per test (can be adjusted easily)
# ---------------------------------------------------------------------
GAIT_PATTERN_IPHONE = "*_2mW_IPhone.xls"   # main 2mW gait file
STS5R_PATTERN = "*_STS5r.xls"
TUG_ST_PATTERN = "*TUGst.xls"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _get_subject_id_from_folder(folder: str) -> str:
    """
    Extract subject id from folder name 'ID_03_T2' → '03'.
    """
    name = os.path.basename(folder.rstrip("/"))
    parts = name.split("_")
    for p in parts:
        if p.isdigit():
            return f"{int(p):02d}"
    raise ValueError(f"Could not parse subject ID from folder: {folder}")


def _find_single_file(folder: str, pattern: str) -> Optional[str]:
    """
    Returns the first file matching `pattern` or None.
    """
    files = sorted(glob(os.path.join(folder, pattern)))
    return files[0] if files else None


def _error_entry(subject_id: str, subject_folder: str, test: str, error: Exception):
    """Return a row dict for the unified error DataFrame."""
    return {
        "id": subject_id,
        "subject_folder": subject_folder,
        "test": test,
        "error": str(error),
    }


# ---------------------------------------------------------------------
# Per-test inference wrappers (per subject)
# ---------------------------------------------------------------------
def infer_gait_imu_for_subject(
    subject_folder: str,
    window_size: int = 256,
    step_size: int = 128,
    sampling_rate_hz: int = 100,
    trajectory_estimation_method: str = "RtsKalman",
    analyze_first_10_strides: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Run 2mW gait analysis for a single subject folder.

    Returns a 1-row DataFrame with an 'id' column and gait features,
    or None if no gait file is found.
    """
    subject_id = _get_subject_id_from_folder(subject_folder)
    gait_file = _find_single_file(subject_folder, GAIT_PATTERN_IPHONE)

    if gait_file is None:
        return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "gait", "file_missing")])


    try:
        df_gait, df_fail = compute_gait_imu_parameters(
            xls_files=[gait_file],
            window_size=window_size,
            step_size=step_size,
            sampling_rate_hz=sampling_rate_hz,
            trajectory_estimation_method=trajectory_estimation_method,
            analyze_first_10_strides=analyze_first_10_strides,
            verbose=False,
        )

        if df_gait.empty:
            return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "gait", "empty_output")])

        # Always enforce 'id' alignment
        df_gait = df_gait.copy()
        df_gait["id"] = subject_id

        # ---- Only keep gait feature columns ----
        expected_cols = [
            "id",
            "mean_stride_length [m]",
            "mean_gait_velocity [m/s]",
            "max_sensor_lift [m]",
            "stride_length_spatial [m]",
            "gait_velocity_spatial [m/s]",
        ]

        # Some subjects might not have spatial results → fill missing with NaN
        for col in expected_cols:
            if col not in df_gait.columns:
                df_gait[col] = np.nan

        df_gait = df_gait[expected_cols]

        # Append any failure logs from gait module
        if df_fail is not None and not df_fail.empty:
            df_fail = df_fail.copy()
            df_fail["id"] = subject_id
            df_fail["subject_folder"] = subject_folder
            df_fail["test"] = "gait"
            return df_gait, df_fail

        return df_gait, pd.DataFrame()

    except Exception as e:
        return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "gait", e)])


def infer_tug_imu_for_subject(
    subject_folder: str,
    acc_sheet_name: str = "Accelerometer",
    window_size: int = 256,
    step_size: int = 128,
    sampling_rate_hz: int = 100,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Run TUG (single-task) IMU analysis for one subject folder.

    Returns a 1-row DataFrame with 'id' and TUG features,
    or None if no TUG file is found.
    """
    subject_id = _get_subject_id_from_folder(subject_folder)
    tug_file = _find_single_file(subject_folder, TUG_ST_PATTERN)

    if tug_file is None:
        return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "tug", "file_missing")])

    try:
        df_tug, df_failed = compute_tug_imu_parameters(
            xls_files=[tug_file],
            acc_sheet_name=acc_sheet_name,
            window_size=window_size,
            step_size=step_size,
            sampling_rate_hz=sampling_rate_hz,
            verbose=verbose,
        )

        if df_tug.empty:
            return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "tug", "empty_output")])

        # Format additional failures from module:
        if df_failed is not None and not df_failed.empty:
            df_failed = df_failed.copy()
            df_failed["id"] = subject_id
            df_failed["subject_folder"] = subject_folder
            df_failed["test"] = "tug"
            return df_tug, df_failed

        return df_tug, pd.DataFrame()

    except Exception as e:
        return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "tug", e)])


def infer_sts_imu_for_subject(
    subject_folder: str,
    label: str = "h",
    fs: float = 100.0,
) -> Optional[pd.DataFrame]:
    """
    Run STS-5r IMU analysis for one subject folder.

    Returns a 1-row DataFrame with 'id' and STS features,
    or None if no STS5r file is found.
    """
    subject_id = _get_subject_id_from_folder(subject_folder)
    sts_file = _find_single_file(subject_folder, STS5R_PATTERN)

    if sts_file is None:
        return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "sts", "file_missing")])

    try:
        df_sts = get_sts_imu_features_single(
            file_path=sts_file,
            label=label,
            fs=fs,
        )
        if df_sts.empty:
            return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "sts", "empty_output")])

        df_sts = df_sts.copy()
        df_sts["id"] = subject_id
        df_sts = df_sts.drop(columns=["subject_id"], errors="ignore")

        return df_sts, pd.DataFrame()

    except Exception as e:
        return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "sts", e)])



def infer_trunk_imu_for_subject(
    subject_folder: str,
    sampling_rate: float = 100.0,
    beta: float = 0.1,
    plot: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Run trunk motion ROM extraction for all flexions (left/right/ant/post)
    for one subject folder.

    Returns a 1-row DataFrame with 'id' and ROM features,
    or None if no trunk files are found.
    """
    subject_id = _get_subject_id_from_folder(subject_folder)
    try:
        df_trunk = compute_all_trunk_motion_for_subject(
            subject_folder=subject_folder,
            sampling_rate=sampling_rate,
            beta=beta,
            plot=plot,
        )
        if df_trunk is None or df_trunk.empty:
            return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "trunk", "empty_output")])
        
        return df_trunk, pd.DataFrame()
    
    except Exception as e:
        return None, pd.DataFrame([_error_entry(subject_id, subject_folder, "trunk", e)])



# ---------------------------------------------------------------------
# High-level subject inference
# ---------------------------------------------------------------------
def infer_subject_features(
    subject_folder: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Main entry point: run ALL available motoric tests for ONE subject.

    Parameters
    ----------
    subject_folder : str
        Path to subject folder, e.g. ".../Termin 2 Vicon/ID_01_T2".
    verbose : bool
        Print progress info.

    Returns
    -------
    full_df : pd.DataFrame
        Wide table with one row for this subject, containing all
        available features (gait, TUG, STS, trunk).
    details : dict[str, DataFrame]
        Per-test DataFrames for debugging/inspection.
        Keys: 'gait', 'tug', 'sts', 'trunk'
    """
    subject_folder = str(subject_folder)
    subject_id = _get_subject_id_from_folder(subject_folder)
    if verbose:
        print(f"\n=== Running inference for subject folder: {subject_folder} ===")

    dfs = []
    details = {}
    errors = []

    # Gait
    gait_df, gait_err = infer_gait_imu_for_subject(subject_folder)
    if gait_df is not None:
        dfs.append(gait_df)
        details["gait"] = gait_df
    if not gait_err.empty:
        errors.append(gait_err)

    # TUG
    tug_df, tug_err = infer_tug_imu_for_subject(subject_folder)
    if tug_df is not None:
        dfs.append(tug_df)
        details["tug"] = tug_df
    if not tug_err.empty:
        errors.append(tug_err)

    # STS
    sts_df, sts_err = infer_sts_imu_for_subject(subject_folder)
    if sts_df is not None:
        dfs.append(sts_df)
        details["sts"] = sts_df
    if not sts_err.empty:
        errors.append(sts_err)

    # Trunk
    trunk_df, trunk_err = infer_trunk_imu_for_subject(subject_folder)
    if trunk_df is not None:
        dfs.append(trunk_df)
        details["trunk"] = trunk_df
    if not trunk_err.empty:
        errors.append(trunk_err)

    # If no features at all → fatal subject failure
    if not dfs:
        raise RuntimeError(f"No valid motoric tests for subject {sid}")

    # Merge all features
    full_df = reduce(lambda l, r: pd.merge(l, r, on="id", how="outer"), dfs)
    cols = ["id"] + [c for c in full_df.columns if c != "id"]
    full_df = full_df[cols]

    # Build full error table
    df_errors = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame()

    return full_df, details, df_errors


# ---------------------------------------------------------------------
# Multi-subject helper
# ---------------------------------------------------------------------
def infer_all_subjects(
    base_folder: str,
    subject_glob: str = "ID_*_T2",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run full inference for ALL subject folders in a base directory.

    Parameters
    ----------
    base_folder : str
        Root path containing subject folders, e.g. DATASETS_02_T2.
    subject_glob : str
        Pattern for subject folders, default "ID_*_T2".
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Big table with one row per subject and all available features.
    """
    base_folder = str(base_folder)
    subject_folders = sorted(glob(os.path.join(base_folder, subject_glob)))

    if not subject_folders:
        raise ValueError(f"No subject folders found in {base_folder} with pattern {subject_glob}")

    all_rows = []
    all_errors = []

    for sf in subject_folders:
        try:
            row_df, _, err_df = infer_subject_features(sf, verbose)
            all_rows.append(row_df)
            if not err_df.empty:
                all_errors.append(err_df)
        except Exception as e:
            # major subject-level failure
            sid = _get_subject_id_from_folder(sf)
            all_errors.append(
                pd.DataFrame([{
                    "id": sid,
                    "subject_folder": sf,
                    "test": "subject",
                    "error": str(e),
                }])
            )
            if verbose:
                print(f"  ⚠ Skipping subject {sid}: {e}")

    if not all_rows:
        raise RuntimeError("No subjects produced valid results.")

    df_all = pd.concat(all_rows, ignore_index=True)
    df_errors = pd.concat(all_errors, ignore_index=True) if all_errors else pd.DataFrame()

    return df_all, df_errors


# ---------------------------------------------------------------------
# CLI / manual testing
# ---------------------------------------------------------------------
if __name__ == "__main__":

    from configs.config import DATASETS_02_T2

    example = os.path.join(DATASETS_02_T2, "ID_01_T2")
    df_one, details_one, err_one = infer_subject_features(example)
    print(df_one.T)
    print("\nErrors:")
    print(err_one)

    df_all, df_err_all = infer_all_subjects(DATASETS_02_T2)
    print(df_all.head())
    print("\nAll errors:")
    print(df_err_all)