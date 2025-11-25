import os 
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R

from src.gait_ml.data.dataset import TrunkDataset
from configs.config import (
    DATASETS_02_T1,
    DATASETS_02_T2,
    )


# ---------------------------------------------------------------------
# 0) Flexion patterns & default axis mapping
# ---------------------------------------------------------------------
FLEXION_PATTERNS = {
    "left": "*lat_l*.xls",
    "right": "*lat_r*.xls",
    "ant": "*ant*.xls",
    "post": "*post*.xls",
}

DEFAULT_AXIS_MAP = {
    "left": "pitch",
    "right": "pitch",
    "ant": "roll",
    "post": "roll",
}


# -----------------------------------------------------------
# 1) Compute Range of Motion (ROM) from IMU orientation
# -----------------------------------------------------------
def get_rom(
    gyro_data: np.ndarray,
    accel_data: np.ndarray,
    flexion_type: str,
    sampling_rate: float,
    axis: str,
    plot: bool = False,
    beta: float = 0.1,
    ):
    """
    Computes spinal ROM using Madgwick orientation estimation.
    Returns ROM and Euler angle traces.
    """
    num_samples = len(accel_data)
    dt = 1.0 / sampling_rate
    
    # World frame gravity vector (Z-axis up)
    g_e = np.array([0, 0, 9.81])

    # -------------------------------------------------------
    # 1. Orientation (Madgwick)
    # -------------------------------------------------------
    madgwick = Madgwick(frequency=sampling_rate, beta=beta) 
    orientations_q = np.zeros((num_samples, 4))
    
    # Initialize the first orientation using the accelerometer
    a_s_initial = accel_data[0]
    rot_align = R.align_vectors(g_e, a_s_initial)[0]
    orientations_q[0] = rot_align.inv().as_quat()

    # Process all subsequent samples
    for k in range(1, num_samples):
        orientations_q[k] = madgwick.updateIMU(
            q=orientations_q[k - 1], gyr=gyro_data[k], acc=accel_data[k]
        )

    # Convert quaternions to rotation objects for easier use
    orientations_R = R.from_quat(orientations_q)
    euler_angles = orientations_R.as_euler('xyz', degrees=True)

    roll = euler_angles[:,0] 
    pitch = euler_angles[:,1]
    yaw = euler_angles[:,2]

    # -------------------------------------------------------
    # 2. Extract movement axis and compute ROM
    # -------------------------------------------------------
    if axis == "roll":
        spine_angles = roll
    elif axis == "pitch":
        spine_angles = pitch
    elif axis == "yaw":
        spine_angles = yaw
    else:
        raise ValueError("Unknown axis")

    # Normalize baseline to zero at start
    baseline = spine_angles[0]
    spine_centered = spine_angles - baseline

    # Use absolute angle to handle left/right symmetry
    abs_signal = np.abs(spine_centered)

    # ROM = peak excursion
    rom = abs_signal.max()
    # -------------------------------------------------------
    # 3. Plot (optional)
    # -------------------------------------------------------  
    if plot:
        #print(f"start: {start}\n end: {end}")
        print(f"ROM: {rom}°")
        plt.figure(figsize=(5, 4))
        plt.plot(spine_angles, label=f"ROM {flexion_type}(°)")
        plt.xlabel("Time (s)" if sampling_rate else "Frame")
        plt.ylabel("deg")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()        
        plt.show()
    return rom,roll, pitch, yaw


# ---------------------------------------------------------------------
# 2) Single-file wrapper
# ---------------------------------------------------------------------
def compute_rom_from_file(
    filepath: str,
    flexion_type: str,
    axis: str | None = None,
    sampling_rate: float = 100.0,
    beta: float = 0.1,
    plot: bool = False,
) -> dict:
    """
    Compute ROM for a single trunk motion file.
    """
    if axis is None:
        if flexion_type not in DEFAULT_AXIS_MAP:
            raise ValueError(f"No default axis for flexion_type='{flexion_type}'")
        axis = DEFAULT_AXIS_MAP[flexion_type]

    result = []

    filepath = str(filepath)
    file_name = Path(filepath).stem
    subject_id = file_name.split("_")[0]

    # Load single-file dataset
    dataset = TrunkDataset(
        xls_files=[filepath],
        flexion_type=flexion_type,
        fs=sampling_rate,
        acc_sheet_name="Accelerometer",
    )

    file_data = dataset.get_file_data(0)
    gyro_data = file_data[:, -3:]
    accel_data = file_data[:, :3]

    rom, roll, pitch, yaw = get_rom(
        gyro_data=gyro_data,
        accel_data=accel_data,
        flexion_type=flexion_type,
        sampling_rate=sampling_rate,
        axis=axis,
        plot=plot,
        beta=beta,
    )

    result.append({
        "id": subject_id,
        "file_name": file_name,
        "flexion_type": flexion_type,
        "axis": axis,
        "rom_deg": rom,
    })
    df = pd.DataFrame(result)
    return df


# ---------------------------------------------------------------------
# 3) Process ALL flexion types for a subject folder
# ---------------------------------------------------------------------
def compute_all_trunk_motion_for_subject(
    subject_folder: str,
    sampling_rate: float = 100.0,
    beta: float = 0.1,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Run ROM extraction for all available flexion types for ONE subject.

    Parameters
    ----------
    subject_folder : str
        Folder path for a subject, e.g. ".../ID_01_T2".
        Expected files:
          - "*lat_l.xls" (left)
          - "*lat_r.xls" (right)
          - "*ant.xls"   (anterior)
          - "*post.xls"  (posterior)
        Missing files are skipped gracefully.
    sampling_rate : float
        IMU sampling frequency.
    beta : float
        Madgwick gain.
    plot : bool
        If True, plot each flexion trace.

    Returns
    -------
    pd.DataFrame
        Wide-format table, one row per subject, columns:
        ['id', 'rom_left_deg', 'rom_right_deg', 'rom_ant_deg', 'rom_post_deg']
        (only columns for flexions actually present).
    """
    subject_folder = str(subject_folder)
    rows = []

    for flexion_type, pattern in FLEXION_PATTERNS.items():
        files = sorted(glob(os.path.join(subject_folder, pattern)))
        if not files:
            # Flexion not recorded → skip
            continue

        # Exactly one file per flexion expected – use first if more.
        row_df = compute_rom_from_file(
            filepath=files[0],
            flexion_type=flexion_type,
            axis=DEFAULT_AXIS_MAP.get(flexion_type),
            sampling_rate=sampling_rate,
            beta=beta,
            plot=plot,
        )
        row = row_df.iloc[0].to_dict() 
        rows.append(row)

    if not rows:
        raise ValueError(f"No trunk motion files found in folder: {subject_folder}")

    df_long = pd.DataFrame(rows)
    # Keep only pivot-relevant columns
    df_long_clean = df_long[["id", "file_name", "flexion_type", "rom_deg"]]

    # Pivot to one row per subject, one column per flexion
    df_wide = df_long_clean.pivot_table(
        index="id",
        columns="flexion_type",
        values="rom_deg",
    )

    # Rename columns to explicit ROM names
    df_wide = df_wide.rename(columns=lambda ft: f"rom_{ft}_deg")
    df_wide.columns.name = None
    df_wide = df_wide.reset_index()

    return df_wide


# -----------------------------------------------------------
# Main execution
# -----------------------------------------------------------
if __name__ == "__main__":

    sampling_rate = 100.0
    axis = "pitch"
    flexion_type = "left"
    beta = 0.1
    plot = False
    
    xls_files =  sorted(glob(os.path.join(DATASETS_02_T2, "ID_*_T2", "*lat_l.xls")))


    dataset_l_h_t2 = TrunkDataset(xls_files=xls_files,
                                flexion_type="left",
                                fs=sampling_rate,
                                acc_sheet_name = "Accelerometer",
                                )
    results = []

    # Loop through files
    for i, path in enumerate(xls_files[:4]):
        file_name = Path(path).stem
        print(f"Processing: {file_name}")

        file_data = dataset_l_h_t2.get_file_data(i) 
        gyro_data  = file_data[:, -3:]
        accel_data = file_data[:, :3]

        rom, roll, pitch, yaw = get_rom(
            gyro_data=gyro_data,
            accel_data=accel_data,
            flexion_type=flexion_type,
            sampling_rate=sampling_rate,
            axis=axis,
            plot=plot,
            beta=beta,
        )
        print(f"ROM → {rom:.2f}°")

        results.append(
            {
                "file_name": file_name,
                "flexion_type": flexion_type,
                "axis": axis,
                "rom_deg": rom,
            }
        )    
    df_res = pd.DataFrame(results)
    print(df_res)