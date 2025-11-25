import os 
import json

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.io as pio
import seaborn as sns

from ahrs.filters import Madgwick
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from scipy.spatial.transform import Rotation
from gaitmap.utils.rotations import rotate_dataset, rotate_dataset_series
from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.zupt_detection import NormZuptDetector, StrideEventZuptDetector
from gaitmap.preprocessing import sensor_alignment
from gaitmap.trajectory_reconstruction import RtsKalman
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory, MadgwickRtsKalman, RtsKalman
from gaitmap.trajectory_reconstruction.orientation_methods import SimpleGyroIntegration
from gaitmap.trajectory_reconstruction.position_methods import ForwardBackwardIntegration

from src.gait_ml.data.dataset import GaitDataset
from src.gait_ml.gait import utils
from configs.config import (
    DATA_PROCESSED, 
    OUTPUT_DIR,
    DATAFRAMES,
    DATASETS_02,
    DATASETS_02_T1,
    DATASETS_02_T2,
    )


def get_gait_features(xls_files, 
                      window_size, 
                      step_size, 
                      sampling_rate_hz, 
                      trajectory_estimation_method="RtsKalman",
                      analyze_first_10_strides=False
                      ):

    all_subjects_results = []
    all_spatial_df = []

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------
    for file_path in xls_files:
        print(f"\n🔹 Processing file: {Path(file_path).name}")

        # Initialize dataset for current file
        dataset = GaitDataset(
            csv_files=[file_path],
            window_size=window_size,
            step_size=step_size,
            expand_labels=0,
        )

        for i in range(len(dataset.raw_x)):
            gyr_data = dataset.raw_x[i][:, :3]
            acc_data = dataset.raw_x[i][:, 3:]
            cur_labels = dataset.raw_y[i]

            # Limit to first ~10 strides for testing
            if analyze_first_10_strides:
                cur_labels[1800:] = 0  # analyze first ~10 strides
                print("WARNING: analyzing only first 10 strides")

            # Gyroscope to degrees
            gyr_data *= 180 / np.pi
            data = pd.DataFrame(np.concatenate([acc_data, gyr_data], axis=1),
                                columns=['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'])
           
            # -----------------------------------------------------
            #  Detect events
            # -----------------------------------------------------
            hs_idx = np.where(cur_labels == 1)[0]
            to_idx = np.where(cur_labels == 2)[0]
            if len(hs_idx) < 2 or len(to_idx) < 2:
                print("⚠️ Not enough stride events detected — skipping.")
                continue

            # Stride list
            df_stride_list = pd.DataFrame([to_idx[:-1], to_idx[1:]]).T - 3
            df_stride_list.columns = ["start", "end"]
            df_stride_list.index.name = "s_id"

            n_strides = len(df_stride_list)
            df_stride_list["ic"] = hs_idx[1:n_strides+1]
            df_stride_list["tc"] = to_idx[:n_strides]
            df_stride_list["min_vel"] = df_stride_list["start"]
            df_stride_list["pre_ic"] = hs_idx[:n_strides]


            R = Rotation.from_matrix([[1, 0, 0],
                                    [0, 0, -1],
                                    [0, 1, 0]])
            data_sf = rotate_dataset(data, R)
            data_sf = sensor_alignment.align_dataset_to_gravity(data_sf, sampling_rate_hz)

            # -----------------------------------------------------
            # --- Trajectory estimation ---
            # -----------------------------------------------------
            if trajectory_estimation_method.lower() == "vanilla":
                trajectory = StrideLevelTrajectory(ori_method=SimpleGyroIntegration(),
                                                pos_method=ForwardBackwardIntegration())
                trajectory.estimate(data=data_sf,
                                    stride_event_list=df_stride_list,
                                    sampling_rate_hz=sampling_rate_hz)
                
            elif trajectory_estimation_method.lower() == "rtskalman":
                trajectory = StrideLevelTrajectory(ori_method=None, 
                                                pos_method=None, 
                                                trajectory_method=RtsKalman(zupt_detector=StrideEventZuptDetector()))
                trajectory.estimate(data=data_sf, 
                                    stride_event_list=df_stride_list, 
                                    sampling_rate_hz=sampling_rate_hz)
            else:
                raise ValueError(f"Unknown trajectory estimation method: {trajectory_estimation_method}")

            # -----------------------------------------------------
            # --- Spatial parameters ---
            # -----------------------------------------------------
            spatial_paras = SpatialParameterCalculation().calculate(
                stride_event_list=df_stride_list,
                positions=trajectory.position_,
                orientations=trajectory.orientation_,
                sampling_rate_hz=sampling_rate_hz,
            )

            cur_df = spatial_paras.parameters_pretty_.round(3)[
                ["gait velocity [m/s]", "stride length [m]", "max. sensor lift [m]"]
            ]
            cur_df["file"] = Path(file_path).stem
            all_spatial_df.append(cur_df)

            # -----------------------------------------------------
            # --- Gait features ---
            # -----------------------------------------------------
            indices = np.nonzero(cur_labels == 1)[0]

            results = utils.calculate_gait_feat_from_imu(acc_data, gyr_data, indices, sampling_rate=sampling_rate_hz)

            mean_stride_length = np.mean(results["stride_lengths"])
            mean_gait_velocity = np.mean(results["gait_velocities"])

            all_subjects_results.append({
                "file": Path(file_path).stem,
                "mean_stride_length": mean_stride_length,
                "mean_gait_velocity": mean_gait_velocity
            })

            print(f"✅ {Path(file_path).name}: "
                f"Stride Length = {mean_stride_length:.2f} m, "
                f"Gait Velocity = {mean_gait_velocity:.2f} m/s")

    # -----------------------------------------------------
    # COMBINE RESULTS
    # -----------------------------------------------------
    df_summary_t1 = pd.DataFrame(all_subjects_results)
    df_spatial_t1 = pd.concat(all_spatial_df, ignore_index=True)

    print("\n✅ All files processed successfully!")
    return df_summary_t1, df_spatial_t1


def compute_gait_imu_parameters(
        xls_files,
        window_size=256,
        step_size=128,
        sampling_rate_hz=100,
        trajectory_estimation_method="RtsKalman",
        analyze_first_10_strides=False,
        verbose=True
    ):
    """
    Unified wrapper for gait IMU feature extraction.
    Returns a single subject-level dataframe + failure logs.

    Parameters
    ----------
    xls_files : list[str]
        List of gait IMU Excel files.
    window_size : int
    step_size : int
    sampling_rate_hz : int
    trajectory_estimation_method : str
        ['RtsKalman', 'vanilla']
    analyze_first_10_strides : bool
    verbose : bool

    Returns
    -------
    df_gait : pd.DataFrame
        Subject-level gait features (stride length, gait velocity, etc.)
    df_fail : pd.DataFrame
        Files that failed during processing.
    """

    # Make sure list
    if isinstance(xls_files, str):
        xls_files = [xls_files]

    results = []
    failures = []

    # Loop through files
    for file_path in xls_files:
        subject_id = Path(file_path).stem.split("_")[0]

        if verbose:
            print(f"\n--- Processing subject {subject_id} ({Path(file_path).name}) ---")

        try:
            # ---- Call existing detailed function ----
            df_summary, df_spatial = get_gait_features(
                [file_path],
                window_size,
                step_size,
                sampling_rate_hz,
                trajectory_estimation_method,
                analyze_first_10_strides
            )

            # df_summary → mean stride length & velocity
            # df_spatial → spatial params from gaitmap

            gait_row = {
                "id": subject_id,
                "file": Path(file_path).name,
            }

            # Add summary features
            if len(df_summary):
                gait_row.update({
                    "mean_stride_length [m]": float(df_summary["mean_stride_length"].values[0]),
                    "mean_gait_velocity [m/s]": float(df_summary["mean_gait_velocity"].values[0]),
                })
            else:
                gait_row.update({
                    "mean_stride_length [m]": np.nan,
                    "mean_gait_velocity [m/s]": np.nan,
                })

            # Add spatial parameters (avg per subject)
            if len(df_spatial):
                gait_row.update({
                    "max_sensor_lift [m]": float(df_spatial["max. sensor lift [m]"].mean()),
                    "stride_length_spatial [m]": float(df_spatial["stride length [m]"].mean()),
                    "gait_velocity_spatial [m/s]": float(df_spatial["gait velocity [m/s]"].mean()),
                })
            else:
                gait_row.update({
                    "max_sensor_lift [m]": np.nan,
                    "stride_length_spatial [m]": np.nan,
                    "gait_velocity_spatial [m/s]": np.nan,
                })

            results.append(gait_row)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            failures.append({
                "id": subject_id,
                "file": file_path,
                "error": str(e)
            })
            continue

    # Final dataframes
    df_gait = pd.DataFrame(results)
    df_fail = pd.DataFrame(failures)

    # Formatting
    if "id" in df_gait.columns:
        df_gait["id"] = df_gait["id"].apply(lambda x: f"{int(x):02d}")

    if verbose:
        print("\n✔ Gait IMU extraction complete.")
        print(f"• Successful: {len(df_gait)} subjects")
        print(f"• Failed: {len(df_fail)}")

    return df_gait, df_fail


if __name__ == "__main__":
    # -----------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------

    xls_files = ["/home/shivamsingh/Projects/SPINE/data/dataset_02/21072025/Termin 1 Vicon/ID_05_T1/05_1_2mW_IPhone.xls",
                "/home/shivamsingh/Projects/SPINE/data/dataset_02/21072025/Termin 1 Vicon/ID_07_T1/07_1_2mW_IPhone.xls",
                "/home/shivamsingh/Projects/SPINE/data/dataset_02/21072025/Termin 1 Vicon/ID_08_T1/08_1_2mW_IPhone.xls",
                "/home/shivamsingh/Projects/SPINE/data/dataset_02/21072025/Termin 1 Vicon/ID_09_T1/09_1_2mW_IPhone.xls",
                #"/home/shivamsingh/Projects/SPINE/data/dataset_02/21072025/Termin 1 Vicon/ID_10_T1/10_1_2mW_IPhone.xls"
                ]
    
    xls_files =  sorted(glob(os.path.join(DATASETS_02_T1, "ID_*_T1", "*_IPhone.xls")))

    window_size = 256
    step_size = 128
    batch_size = 8
    sampling_rate_hz = 100
    trajectory_estimation_method = "RtsKalman"
    analyze_first_10_strides = False

    df_summary_t1, df_spatial_t1 = get_gait_features(xls_files, 
                                                    window_size, 
                                                    step_size, 
                                                    sampling_rate_hz, 
                                                    trajectory_estimation_method=trajectory_estimation_method,
                                                    analyze_first_10_strides=analyze_first_10_strides
                                                    )
    print("\nFinal Summary:")
    print(df_summary_t1)