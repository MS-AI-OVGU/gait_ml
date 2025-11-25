import os 
import json
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.signal import find_peaks, butter, sosfiltfilt, savgol_filter, peak_widths
from scipy.spatial.transform import Rotation as R

from src.gait_ml.data.dataset import SitToStandDataset
from configs.config import (
    DATA_PROCESSED, 
    OUTPUT_DIR,
    DATAFRAMES,
    DATASETS_02,
    DATASETS_02_T1,
    DATASETS_02_T2,
    )


# --------------------------------------------------------------------------
# 01: Baseline detection
# --------------------------------------------------------------------------
def detect_baseline_end(signal, win=100, var_th=0.6, amp_th=5):

    var_series = np.array([np.var(signal[i:i+win]) for i in range(len(signal)-win)])    
    # candidate start = first window above variance threshold
    candidate = np.argmax(var_series > var_th)    
    # refine: make sure the signal amplitude after this point is big enough
    for i in range(candidate, len(signal)):
        if abs(signal[i]) > amp_th:   # subject really moving
            return i
    return 0


# --------------------------------------------------------------------------
# 02: Phase segmentation
# --------------------------------------------------------------------------
def segment_sts_by_triplet(signal, fs=100, plot=True):
    """Segment Sit→Stand and Stand→Sit phases from gyro signal."""

    sig = signal.copy()
    sig[0] = 0  # ensure first crossing is caught
    zc_points = np.where(np.diff(np.sign(sig)))[0]

    phases = []
    for i in range(0, len(zc_points) - 2, 2):  
        start, mid, end = zc_points[i], zc_points[i+1], zc_points[i+2]
        phase_type = "Sit→Stand" if (len(phases) % 2 == 0) else "Stand→Sit"
        phases.append({
            "start_idx": int(start),
            "end_idx": int(end),
            "start_idx_new": int(start),
            "end_idx_new": int(end),
            "mid_idx": int(mid),
            "phase_type": phase_type,
            "duration_s": (end - start) / fs
        })
    df = pd.DataFrame(phases)

    if plot:
        plt.figure(figsize=(12,5))
        plt.plot(sig, label="GyroX")
        plt.axhline(0, ls="--", c="gray")
        for p in phases:
            color = "green" if p["phase_type"] == "Sit→Stand" else "red"
            plt.axvspan(p["start_idx"], p["end_idx"], alpha=0.3, color=color, label=p["phase_type"])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title("STS Phase Segmentation (Zero-Cross Triplets)")
        plt.show()
    return df, phases


# --------------------------------------------------------------------------
# 03: Feature computation
# --------------------------------------------------------------------------
def compute_sts_imu_features(phases, accel, gyro, fs=100):
    """
    Compute STS features from segmented phases.
    Includes jerk metrics.
    """
    rep_durations = []
    peak_acc_list = []
    peak_gyr_list = []
    
    sit_acc_means = []
    sit_gyr_means = []
    stand_acc_means = []
    stand_gyr_means = []
    
    jerk_vals = []   # per-phase jerk
    
    # --- Group 2 phases into 1 repetition ---
    for i in range(0, len(phases), 2):
        if i+1 >= len(phases): break
        p1, p2 = phases[i], phases[i+1]
        
        rep_duration = p1["duration_s"] + p2["duration_s"]
        rep_durations.append(rep_duration)
        
        # signals
        acc_seg = accel[p1["start_idx"]:p2["end_idx"]]
        gyr_seg = gyro[p1["start_idx"]:p2["end_idx"]]
        
        # peaks
        peak_acc_list.append(np.max(acc_seg))
        peak_gyr_list.append(np.max(gyr_seg))
    
    # --- Per-phase stats (acc, gyr, jerk) ---
    sit2stand_durations = []
    stand2sit_durations = []
    
    for p in phases:
        acc_seg = accel[p["start_idx"]:p["end_idx"]]
        gyr_seg = gyro[p["start_idx"]:p["end_idx"]]

        # jerk (discrete derivative of acc magnitude)
        jerk = np.diff(acc_seg) * fs
        jerk_rms = np.sqrt(np.mean(jerk**2)) if len(jerk) > 0 else 0
        jerk_vals.append(jerk_rms)
        
        if p["phase_type"] == "Sit→Stand":
            sit_acc_means.append(np.mean(acc_seg))
            sit_gyr_means.append(np.mean(gyr_seg))
            sit2stand_durations.append(p["duration_s"])
        else:
            stand_acc_means.append(np.mean(acc_seg))
            stand_gyr_means.append(np.mean(gyr_seg))
            stand2sit_durations.append(p["duration_s"])
    
    # --- Aggregate features ---
    features = {}
    features["STS_total_time"] = sum(rep_durations)
    # Phase-level temporal
    features["sit2stand_average_time"] = np.mean(sit2stand_durations) 
    features["stand2sit_average_time"] = np.mean(stand2sit_durations) 
    features["sit2stand_time_SD"] = np.std(sit2stand_durations, ddof=1) 
    features["stand2sit_time_SD"] = np.std(stand2sit_durations, ddof=1)
    features["sit2stand_variability_time"] = 100 * features["sit2stand_time_SD"] / features["sit2stand_average_time"] 
    features["stand2sit_variability_time"] = 100 * features["stand2sit_time_SD"] / features["stand2sit_average_time"] 
    features["time_asymmetry"] = (features["sit2stand_average_time"] - features["stand2sit_average_time"]) / features["sit2stand_average_time"] * 100

    # Rep-level temporal
    features["Rep_duration_avg"] = np.mean(rep_durations)
    features["Rep_Time_SD"] = np.std(rep_durations)
    features["Rep_variability_time"] = 100 * np.std(rep_durations) / np.mean(rep_durations)
    features["Rep_count"] = len(rep_durations)
    features["Rep_variability_acc"] = 100 * np.std(peak_acc_list) / np.mean(peak_acc_list)
    features["Rep_variability_angVel"] = 100 * np.std(peak_gyr_list) / np.mean(peak_gyr_list)
    
    features["SitToStand_Average_Acceleration"] = np.mean(sit_acc_means)
    features["StandToSit_Average_Acceleration"] = np.mean(stand_acc_means)
    features["STS_Acceleration_Asymmetry"] = 100 * abs(np.mean(sit_acc_means)-np.mean(stand_acc_means)) / np.mean([*sit_acc_means,*stand_acc_means])
    
    features["SitToStand_Average_AngVel"] = np.mean(sit_gyr_means)
    features["StandToSit_Average_AngVel"] = np.mean(stand_gyr_means)
    features["STS_Angular_Velocity_Asymmetry"] = 100 * abs(np.mean(sit_gyr_means)-np.mean(stand_gyr_means)) / np.mean([*sit_gyr_means,*stand_gyr_means])
    
    features["Mean_acceleration"] = np.mean([*sit_acc_means,*stand_acc_means])
    features["Peak_acceleration"] = np.max(peak_acc_list)
    features["Mean_angular_velocity"] = np.mean([*sit_gyr_means,*stand_gyr_means])
    features["Peak_angular_velocity"] = np.max(peak_gyr_list)
    
    # --- Jerk features ---
    features["Mean_jerk_rms"] = np.mean(jerk_vals)
    features["Peak_jerk_rms"] = np.max(jerk_vals)
    features["Rep_variability_jerk"] = 100 * np.std(jerk_vals) / np.mean(jerk_vals)

    features = {k: round(v, 2) if isinstance(v, (int, float, np.floating)) else v for k, v in features.items()}
   
    return features


# --------------------------------------------------------------------------
# 04: Main function to extract features for all subjects
# --------------------------------------------------------------------------
def get_sts_imu_features(sts_files, label="h", fs=100.0):
    """Extract IMU STS features for all subjects."""

    dataset_sts = SitToStandDataset(xls_files=sts_files, 
                                    label=label,
                                    acc_sheet_name="Linear Accelerometer")
    subject_results = []
    for idx, file_path in enumerate(sts_files):
        # --- Extract subject_id from path (e.g., 'ID_03_T2')
        subject_id = int(os.path.basename(os.path.dirname(file_path)).split("_")[1])

        # --- Get preprocessed IMU data for this subject
        imu_data = dataset_sts.get_file_data(idx)

        # --- Compute signal magnitudes
        gyro_mag = np.linalg.norm(imu_data[:, 3:], axis=1)
        accel_mag = np.linalg.norm(imu_data[:, :3], axis=1)
        gyro_x = imu_data[:, 3]

        # --- Detect baseline (motion start)
        imu_t0 = detect_baseline_end(gyro_x, win=100, var_th=0.6, amp_th=5)
        imu_aligned = gyro_x[imu_t0:]

        # --- Segment phases
        _, phases = segment_sts_by_triplet(imu_aligned, plot=False)
        # --- Compute features
        features = compute_sts_imu_features(phases, accel_mag, gyro_mag, fs)

        # --- Add metadata        
        features["subject_id"] = subject_id
        features["file_path"] = file_path
        subject_results.append(features)
    return pd.DataFrame(subject_results)


# --------------------------------------------------------------------------
# 05: extract features for a single subject
# --------------------------------------------------------------------------
def get_sts_imu_features_single(file_path, label="h", fs=100.0):
    """
    Extract IMU-based Sit-to-Stand (STS) features for a single subject file.

    Parameters
    ----------
    file_path : str
        Path to the subject's IMU Excel file.
    label : str, optional
        Subject label or condition (e.g., 'h', 'p'), by default "h".
    fs : float, optional
        Sampling frequency in Hz, by default 100.0.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame containing the computed IMU STS features.
    """

    # --- Load dataset (single file wrapped in list)
    dataset_sts = SitToStandDataset(
        xls_files=[file_path],
        label=label,
        acc_sheet_name="Linear Accelerometer"
    )

    # --- Extract subject_id from path (e.g., 'ID_03_T2')
    subject_id = int(os.path.basename(os.path.dirname(file_path)).split("_")[1])

    # --- Get preprocessed IMU data for this subject
    imu_data = dataset_sts.get_file_data(0)   # only one file

    # --- Compute signal magnitudes
    gyro_mag = np.linalg.norm(imu_data[:, 3:], axis=1)    # deg/s
    accel_mag = np.linalg.norm(imu_data[:, :3], axis=1)   # m/s²
    gyro_x = imu_data[:, 3]                               # deg/s

    # --- Detect baseline (motion start)
    imu_t0 = detect_baseline_end(gyro_x, win=100, var_th=0.6, amp_th=5)
    imu_aligned = gyro_x[imu_t0:]

    # --- Segment phases
    _, phases = segment_sts_by_triplet(imu_aligned, plot=False)
    # --- Compute features
    features = compute_sts_imu_features(phases, accel_mag, gyro_mag, fs)

    # --- Add metadata
    features["subject_id"] = subject_id
    features["file_path"] = file_path
    features["label"] = label
    return pd.DataFrame([features])



if __name__ == "__main__":
    #DRAFT
    sts5r_h_xls_files_t2 = sorted(glob(os.path.join(DATASETS_02_T2,"ID_*_T2", "*_STS5r.xls"))) #dataset_02 ; subject=01_2_STS5r
    print(sts5r_h_xls_files_t2[0])


    cutoff=0.6
    fs=100.0

    dataset_sts_h_t2 = SitToStandDataset(
        xls_files=sts5r_h_xls_files_t2, 
        label="h",
        acc_sheet_name="Linear Accelerometer"
        )
    
    idx = 0
    gyro_mag  = np.linalg.norm(dataset_sts_h_t2.get_file_data(idx)[:,3:], axis=1)
    accel_mag = np.linalg.norm(dataset_sts_h_t2.get_file_data(idx)[:,:3], axis=1)

    # load imu
    gyrX_smooth = dataset_sts_h_t2.get_file_data(idx)[:,3]
    imu_signal = gyrX_smooth * 180.0 / np.pi

    #load vicon
    #vicon_signal =  model_df["01RPelvisAngles_X'"].to_numpy()

    #detect baseline
    #vicon_t0 = detect_baseline_end(vicon_signal, win=100, var_th=0.6, amp_th=5)
    imu_t0   = detect_baseline_end(imu_signal, win=100, var_th=0.6, amp_th=5)
    imu_aligned   = imu_signal[imu_t0:]

    # mean_start = vicon_t0 + imu_t0
    # mean_start = mean_start // 2
    # offset = 10
    # vicon_aligned = vicon_signal[mean_start-offset:]
    # imu_aligned   = imu_signal[mean_start-offset:]

    df_sts_phase, phases = segment_sts_by_triplet(imu_aligned, plot=False)
    df_sts_phase

    imu_gyro_mag = gyro_mag * 180.0 / np.pi
    features_imu_sts = compute_sts_imu_features(phases, accel_mag, imu_gyro_mag, fs=100)
    #vicon_sts_feat


    imu_sts_feat = pd.DataFrame([features_imu_sts])
    imu_sts_feat.T

