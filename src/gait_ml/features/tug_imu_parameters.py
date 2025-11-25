import os
import json

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.gait_ml.data.dataset import GaitDataset
from src.gait_ml.gait import utils
from configs.config import DATASETS_02_T2


def compute_tug_imu_parameters(
        xls_files,
        acc_sheet_name="Accelerometer",
        window_size=256,
        step_size=128,
        sampling_rate_hz=100,
        verbose=True,
    ):
    """
    Compute TUG IMU parameters (temporal + kinematic) for one or multiple subjects.

    Parameters
    ----------
    xls_files : str or list
        Glob pattern or list of .xls file paths.
    acc_sheet_name : str
        Name of the accelerometer sheet in Excel.
    window_size : int
        Window size for dataset loader.
    step_size : int
        Step size for dataset loader.
    sampling_rate_hz : int
        Sampling rate of IMU.
    verbose : bool
        Print subject-level debug output.

    Returns
    -------
    df_tug_feat : pd.DataFrame
        Full TUG feature table (temporal + kinematic), one row per subject.
    """

    # ---------------------------------------------------
    # 1. Resolve file list
    # ---------------------------------------------------
    if isinstance(xls_files, str):
        xls_files = sorted(glob(xls_files))

    if len(xls_files) == 0:
        raise ValueError("No TUG files found for the provided path/pattern.")

    if verbose:
        print(f"Found {len(xls_files)} TUG files.")

    # ---------------------------------------------------
    # 2. Load dataset
    # ---------------------------------------------------
    dataset = GaitDataset(
        csv_files=xls_files,
        window_size=window_size,
        step_size=step_size,
        expand_labels=0,
        acc_sheet_name=acc_sheet_name,
    )

    # Extract subject IDs
    subj_ids = [Path(path).stem.split("_")[0] for path in xls_files]

    temporal_feat = []
    kinematic_feat = []
    failed = []

    # ---------------------------------------------------
    # 3. Compute features per subject
    # ---------------------------------------------------
    for k, raw in enumerate(dataset.raw_x):

        subject = subj_ids[k]
        file_path = xls_files[k]
        if verbose:
            print(f"\nProcessing subject: {subject}")

        try:
            # Split into gyro + acc (first 3 acc, last 3 gyro)
            gyr_data = raw[:, :3]
            acc_data = raw[:, 3:]

            # Add dummy axis for SPINE (shape must be 4xN)
            gyr_data = np.column_stack([gyr_data, np.zeros(len(gyr_data))])
            acc_data = np.column_stack([acc_data, np.zeros(len(acc_data))])

            # ---------------------------------------------------
            # Detect events & compute temporal features
            # ---------------------------------------------------
            events = utils.detect_tug_events(acc_data.T, gyr_data.T)
            tug_temporal = utils.calculate_tug_phase_durations(events)
            tug_temporal["id"] = subject
            temporal_feat.append(tug_temporal)

            # ---------------------------------------------------
            # Compute kinematic features
            # ---------------------------------------------------
            tug_kin = utils.calculate_kinematic_features(
                events,
                acc_data[:, :3],   # real 3-axis acc
                gyr_data[:, :3],   # real 3-axis gyro
            )
            tug_kin["id"] = subject
            kinematic_feat.append(tug_kin)

        except Exception as E:
            error_msg = str(E)
            print(f"❌ Error processing {subject}: {E}")
            failed.append({
                "id": subject,
                "file": file_path,
                "error": error_msg
            })
            continue

    # ---------------------------------------------------
    # 4. Build final DataFrames and merge
    # ---------------------------------------------------
    df_temporal = pd.DataFrame(temporal_feat)
    df_kinematic = pd.DataFrame(kinematic_feat)

    # Enforce ID formatting (e.g., "01", "02")
    df_temporal["id"] = df_temporal["id"].apply(lambda x: f"{int(x):02d}")
    df_temporal.insert(0, "id", df_temporal.pop("id"))
    df_kinematic["id"] = df_kinematic["id"].apply(lambda x: f"{int(x):02d}")

    # Merge temporal + kinematic
    df_tug_feat = df_temporal.merge(df_kinematic, on="id", how="inner")

    if verbose:
        print("\n✔️ TUG feature extraction complete.")
        print(f"Subjects processed: {len(df_tug_feat)}")
        print(f"Subjects failed: {len(failed)}")

    return df_tug_feat,  pd.DataFrame(failed)


if __name__ == "__main__":

    # Example usage: For all subjects
    xls_files=os.path.join(DATASETS_02_T2, "ID_*_T2", "*TUGst.xls")

    df_tug_features, df_failed = compute_tug_imu_parameters(
        xls_files=xls_files,
        acc_sheet_name="Accelerometer",
        window_size=256,
        step_size=128,
        sampling_rate_hz=100,
        verbose=True,
    )
    print(df_tug_features)

    # Example usage: For one subject
    # xls_file = "/data/.../ID_05_T2/*TUGst.xls",
    # df_tug, df_failed = compute_tug_imu_parameters(
    #       xls_files=xls_file,
    #       acc_sheet_name="Accelerometer",
    #       window_size=256,
    #       step_size=128,
    #       sampling_rate_hz=100,
    #       verbose=True
    # )
    # print(df_tug)
    # 


    