import ahrs
import matplotlib.pyplot as plt
import numpy as np
from ahrs.filters import Madgwick
from scipy.signal import butter, find_peaks, peak_widths, sosfiltfilt
from scipy.spatial.transform import Rotation
from ahrs.filters import EKF


def detect_turns(accel_mag, gyro_mag, accel_thresh=None, gyro_thresh=None):
    # Calculate thresholds automatically if not provided
    if accel_thresh is None or gyro_thresh is None:
        accel_thresh, gyro_thresh = calculate_adaptive_thresholds(accel_mag, gyro_mag)

    walking_times = []
    turning_times = []

    is_turning = False
    turn_start = None
    walk_start = None

    for i in range(len(accel_mag)):
        if accel_mag[i] < accel_thresh and gyro_mag[i] > gyro_thresh:  # Turning phase
            if not is_turning:  # If we are switching to a turn
                is_turning = True
                if walk_start is not None:  # If a walking phase just ended
                    walking_times.append(i - walk_start)
                turn_start = i
        else:  # Walking phase
            if is_turning:  # If we were turning and now switching back to walking
                is_turning = False
                turning_times.append(i - turn_start)
                walk_start = i  # Start a new walking phase

    # Edge case: If the last phase was walking or turning, add it
    if is_turning and turn_start is not None:
        turning_times.append(len(accel_mag) - turn_start)
    elif not is_turning and walk_start is not None:
        walking_times.append(len(accel_mag) - walk_start)

    print("Walking times (samples):", walking_times)
    print("Turning times (samples):", turning_times)

    # Convert to seconds
    sampling_rate = 100

    walking_times = [t / sampling_rate for t in walking_times]
    turning_times = [t / sampling_rate for t in turning_times]

    print("Walking Times (seconds):", walking_times)
    print("Turning Times (seconds):", turning_times)

    return walking_times, turning_times


def calculate_mad(data):
    """Calculate Median Absolute Deviation (MAD)"""
    median = np.median(data)
    deviations = np.abs(data - median)
    mad = np.median(deviations)
    return mad


def adaptive_threshold_mad(data, threshold_factor=5.3):
    """Calculate an adaptive threshold using MAD method"""
    mad = calculate_mad(data)
    median = np.median(data)
    threshold = median + threshold_factor * mad
    return threshold


def calculate_adaptive_thresholds(accel_mag, gyro_mag):
    """Calculate separate adaptive thresholds for acceleration and gyroscope"""
    accel_thresh = adaptive_threshold_mad(accel_mag)
    gyro_thresh = adaptive_threshold_mad(gyro_mag)
    return accel_thresh, gyro_thresh


def estimate_total_distance_v2(total_time, t_straight, t_turn):
    # Calculate the duration of one full cycle (T_straight + T_turn)
    cycle_duration = t_straight + t_turn

    # Calculate the total number of full cycles completed during the total time
    num_cycles = total_time / cycle_duration

    # Total distance per cycle (10m forward, 1.04m turn, 10m backward, 1.04m turn)
    cycle_distance = 10 + 1.04 + 10 + 1.04  # 22.08 meters

    # Calculate the total distance walked
    total_distance = num_cycles * cycle_distance

    return total_distance


def calculate_gait_feat_from_imu(
    accel_data: np.ndarray,  # in m/s^2
    gyro_data: np.ndarray,  # in rad/s
    stance_indices: list,  # Indices of stance events (e.g., heel strikes)
    sampling_rate: float,
) -> dict:
    """
    Calculates the 3D trajectory and stride lengths from shank-mounted IMU data.

    This function implements a full strapdown integration pipeline including:
    1. Madgwick filter for robust orientation estimation.
    2. Double integration of acceleration to get position.
    3. Zero-Velocity Update (ZUPT) for velocity drift correction.
    """
    num_samples = len(accel_data)
    dt = 1.0 / sampling_rate

    # World frame gravity vector (Z-axis up)
    g_e = np.array([0, 0, 9.81])

    # -------------------------------------------------------------------------
    # Step 1: Determine Sensor Orientation (Madgwick Filter)
    # -------------------------------------------------------------------------
    madgwick = Madgwick(frequency=sampling_rate, beta=0.1)
    orientations_q = np.zeros((num_samples, 4))

    # Initialize the first orientation using the accelerometer
    a_s_initial = accel_data[0] / np.linalg.norm(accel_data[0])
    rot_align = Rotation.align_vectors(g_e, a_s_initial)[0]
    orientations_q[0] = rot_align.inv().as_quat(scalar_first=True)
    # orientations_q[0] = [1, 0, 0, 0]  # Temporary fix for initial orientation

    # Process all subsequent samples
    for k in range(1, num_samples):
        orientations_q[k] = madgwick.updateIMU(
            q=orientations_q[k - 1], gyr=gyro_data[k], acc=accel_data[k]
        )

    # Convert quaternions to rotation objects for easier use
    Q_scipy = orientations_q[:, [1, 2, 3, 0]]
    orientations_R = Rotation.from_quat(Q_scipy)

    # -------------------------------------------------------------------------
    # Step 2: Calculate Linear Acceleration in the World Frame
    # -------------------------------------------------------------------------
    # Rotate sensor acceleration into the world frame and subtract gravity
    linear_accel_e = orientations_R.apply(accel_data) - g_e

    # -------------------------------------------------------------------------
    # Step 3: Integrate Acceleration to Find Raw (Drifting) Velocity
    # -------------------------------------------------------------------------
    raw_velocity_e = np.zeros((num_samples, 3))
    for k in range(1, num_samples):
        raw_velocity_e[k] = (
            raw_velocity_e[k - 1]
            + 0.5 * (linear_accel_e[k] + linear_accel_e[k - 1]) * dt
        )

    # -------------------------------------------------------------------------
    # Step 4: Correct Velocity Drift with Zero-Velocity Updates (ZUPT)
    # -------------------------------------------------------------------------
    corrected_velocity_e = np.copy(raw_velocity_e)

    for i in range(len(stance_indices) - 1):
        start_k = stance_indices[i]
        end_k = stance_indices[i + 1]

        duration_samples = end_k - start_k
        if duration_samples == 0:
            continue

        # The velocity at each stance event should be zero.
        # The value in raw_velocity_e at these points is the accumulated drift.
        v_drift_start = raw_velocity_e[start_k]
        v_drift_end = raw_velocity_e[end_k]

        # Create a linear drift correction ramp for this stride
        drift_correction = np.linspace(v_drift_start, v_drift_end, duration_samples + 1)

        # Subtract the drift from the velocity profile of this stride
        corrected_velocity_e[start_k : end_k + 1] -= drift_correction
    # return raw_velocity_e, corrected_velocity_e, drift_correction
    # -------------------------------------------------------------------------
    # Step 5: Integrate Corrected Velocity to Find Trajectory
    # -------------------------------------------------------------------------
    positions_e = np.zeros((num_samples, 3))
    for k in range(1, num_samples):
        positions_e[k] = (
            positions_e[k - 1]
            + 0.5 * (corrected_velocity_e[k] + corrected_velocity_e[k - 1]) * dt
        )
    # # --- Step 5b: Compensate for Sensor Position to get Ankle Trajectory ---
    # r_imu_to_ankle_s = np.array([0.0, 0.0, -0.40]) # EXAMPLE: Measure your own
    # r_imu_to_ankle_e = orientations_R.apply(r_imu_to_ankle_s)
    # positions_e = positions_e + r_imu_to_ankle_e

    # -------------------------------------------------------------------------
    # Step 6: Calculate Stride Length from the Trajectory
    # -------------------------------------------------------------------------
    stride_lengths = []
    gait_velocities = []
    positions = []
    for i in range(len(stance_indices) - 1):
        start_k = stance_indices[i]
        end_k = stance_indices[i + 1]

        start_pos = positions_e[start_k]
        end_pos = positions_e[end_k]

        # Calculate Euclidean distance between the start and end of the stride
        displacement = end_pos - start_pos
        stride_length = np.linalg.norm(displacement)
        stride_lengths.append(stride_length)

        stride_duration = (end_k - start_k) * dt
        if stride_duration > 0:
            gait_velocity = stride_length / stride_duration
            gait_velocities.append(gait_velocity)
        else:
            gait_velocities.append(0)  # Avoid division by zero
        positions.append(positions_e)
    return {
        "trajectory": positions,
        # "corrected_velocity": corrected_velocity_e,
        "stride_lengths": stride_lengths,
        "gait_velocities": gait_velocities,
        # "orientations": orientations_R
    }


# TUG
# def _get_highest_peak(sig, th=None, fs=100):
#     peaks, _ = find_peaks(sig)
#     highest_peak_idx = peaks[np.argmax(sig[peaks])]
#     if th is None:
#         return highest_peak_idx
#     else:
#         high_peak_th = th * sig[highest_peak_idx]
#         high_peaks, _ = find_peaks(sig, height=high_peak_th, distance=fs // 2)
#         return high_peaks


# def _get_zero_crossing(sig):
#     zero_cross = []
#     for i in range(len(sig)):
#         if i == len(sig) - 1:
#             break
#         if np.sign(sig[i]) != np.sign(sig[i + 1]):
#             zero_cross.append(i)

#     return np.array(zero_cross)


def _get_highest_peak(sig: np.ndarray, th: float | None = None, fs: int = 100):
    sig = np.asarray(sig)
    if sig.size == 0:
        return None if th is None else np.array([], dtype=int)
    peaks, _ = find_peaks(sig)
    if len(peaks) == 0:
        return None if th is None else np.array([], dtype=int)
    # highest peak index (in `sig`)
    highest_peak = peaks[np.argmax(sig[peaks])]
    if th is None:
        return int(highest_peak)
    # thresholded set of peaks around the highest one
    high_peak_th = th * sig[highest_peak]
    high_peaks, _ = find_peaks(sig, height=high_peak_th, distance=fs // 2)
    # If thresholding kills everything, keep at least the global highest peak
    if len(high_peaks) == 0:
        high_peaks = np.array([highest_peak], dtype=int)

    return high_peaks.astype(int)


def _get_zero_crossing(sig: np.ndarray) -> np.ndarray:
    """Return indices where the signal changes sign."""
    sig = np.asarray(sig)
    if sig.size < 2:
        return np.array([], dtype=int)

    sgn = np.sign(sig)
    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    return idx.astype(int)


def detect_tug_events(acc: np.ndarray, gyr: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    Detect and label events during a Timed Up and Go (TUG) test using IMU data.

    This function identifies seven key events in the TUG test using gyroscope
    and accelerometer signals from a lower-back mounted IMU sensor.

    Parameters:
    -----------
    acc : np.ndarray
        Accelerometer data with shape (3, n_samples) representing x, y, z axes
    gyr : np.ndarray
        Gyroscope data with shape (3, n_samples) representing x, y, z axes

    Returns:
    --------
    events : np.ndarray
        Array of same length as input signals with events labeled as:
        0: No event
        1: Sit to Stand start
        2: Sit to Stand end / Walk forward start
        3: Turn 1 start (Turn around cone)
        4: Turn 1 end / Walk back start
        5: Turn 2 start (Turn before sitting)
        6: Turn 2 end / Stand to Sit start
        7: Stand to Sit end

    Notes:
    ------
    Event detection process:
    1. Applies lowpass filter (1Hz cutoff) to reduce noise
    2. Uses gyroscope X signal for sit-to-stand transitions (events 1,2)
    3. Uses gyroscope Y signal squared for turn detection (events 3,4,5)
    4. Uses peak detection and zero crossings to identify event boundaries
    5. Sampling rate is assumed to be 100 Hz

    The function asserts that all 7 events are detected in sequence.

    Example:
    --------
    >>> events = detect_tug_events(accelerometer_data, gyroscope_data)
    >>> event_times = np.where(events != 0)[0] / 100.0  # Convert to seconds
    """
    # Initialize lowpass filter (1 Hz cutoff, 100 Hz sampling)
    sos = butter(10, 1, btype="lowpass", fs=fs, output="sos")

   # Extract individual axes
    gyroX = gyr[0, :]  # Mainly captures anterior-posterior rotation
    gyroY = gyr[1, :]  # Mainly captures mediolateral rotation
    gyroZ = gyr[2, :]  # Mainly captures vertical rotation
    accY = acc[1, :]  # Anterior-posterior acceleration

    # Initialize events array
    N = gyroX.size
    events = np.zeros(N, dtype=int)

    # ------------------------------------------------------------------
    # 1) Turn peaks (for trunk_ind & intermediate turn)
    # ------------------------------------------------------------------
    # Calculate squared gyroY signal for turn detection
    pre_gyroY = np.power(gyroY, 2)

    pre_gyroY_peaks = _get_highest_peak(pre_gyroY, th=0.4, fs=fs)
    if pre_gyroY_peaks is None or pre_gyroY_peaks.size == 0:
        raise RuntimeError("No turning peaks found in gyroY – cannot detect TUG events.")

    # ------------------------------------------------------------------
    # 2) Sit-to-stand start (event 1)
    # ------------------------------------------------------------------
    # Detect Sit-to-Stand initiation (Event 1)
    min_ind = int(np.argmin(gyroX))
    width_info = peak_widths(np.abs(gyroX), [min_ind], rel_height=0.9)
    rising_ind = int(width_info[2])
    events[rising_ind] = 1

    # Detect Stand-to-Sit initiation (Event 6)
    # Assume standing to sitting occurs after the last turn + 100ms to handle unrealistic angles
    # last strong turn peak → roughly where subject starts to prepare to sit
    # trunk_ind = pre_gyroY_peaks[-1] + 20
    trunk_ind = int(pre_gyroY_peaks[-1] + 0.2 * fs)  # +200 ms instead of fixed 20 samples    
    trunk_ind = min(trunk_ind, N - 1)
    events[trunk_ind] = 6  # Stand-to-sit start (event 6)

    # ------------------------------------------------------------------
    # 3) Detect Final turn before sitting (event 5)
    # ------------------------------------------------------------------
    zero_ind_all = _get_zero_crossing(gyroY)
    zero_before_trunk = zero_ind_all[zero_ind_all < trunk_ind]
    if zero_before_trunk.size == 0:
        raise RuntimeError("No gyroY zero-crossing before trunk_ind – cannot find event 5.")
    fin_rot_ind = int(zero_before_trunk[-1])
    events[fin_rot_ind] = 5

    # ------------------------------------------------------------------
    # 4) Intermediate turn (events 3 & 4)
    # ------------------------------------------------------------------
    pre_gyroY_int = sosfiltfilt(sos, np.power(gyroY[:fin_rot_ind], 2))
    int_rot_peak = _get_highest_peak(pre_gyroY_int)
    if int_rot_peak is None:
        raise RuntimeError("No intermediate turn peak in gyroY – cannot find events 3/4.")
    rot_width_info = peak_widths(pre_gyroY_int, [int_rot_peak], rel_height=0.9)
    left_rot = int(rot_width_info[2])   # event 3: turn-1 start
    right_rot = int(rot_width_info[3])  # event 4: turn-1 end
    events[left_rot] = 3
    events[right_rot] = 4

    # ------------------------------------------------------------------
    # 5) Sit-to-stand completion (event 2)
    # ------------------------------------------------------------------

    if right_rot <= rising_ind:
        raise RuntimeError("Turn-1 occurs before sit-to-stand – check signal / labels.")
    
    # Detect Sit-to-Stand completion (Event 2)
    sig = gyroX[rising_ind:right_rot]
    if sig.size == 0:
        raise RuntimeError("Empty segment for sit-to-stand completion.")

    rising_peak_idx = _get_highest_peak(sig)
    if rising_peak_idx is None:
        raise RuntimeError("No peak in sit-to-stand segment – cannot locate event 2.")

    zero_after_peak = _get_zero_crossing(sig[rising_peak_idx:])
    if zero_after_peak.size > 0:
        rising_zero_rel = int(zero_after_peak[0])
    else:
        # fallback: closest-to-zero sample after the peak
        rising_zero_rel = int(np.argmin(np.abs(sig[rising_peak_idx:])))

    ev2_idx = rising_ind + rising_peak_idx + rising_zero_rel
    ev2_idx = min(ev2_idx, N - 1)
    events[ev2_idx] = 2

    # ------------------------------------------------------------------
    # 6) Final sitting (event 7)
    # ------------------------------------------------------------------
    sig_tail = gyroX[trunk_ind:]
    if sig_tail.size == 0:
        raise RuntimeError("No samples after trunk_ind – cannot locate final sitting.")

    peak_tail_idx = _get_highest_peak(sig_tail)
    if peak_tail_idx is None:
        # fallback: just use trunk_ind as start of sit-down flexion
        peak_tail_idx = 0
    x_peak_in = trunk_ind + peak_tail_idx

    zc_after = _get_zero_crossing(gyroX[x_peak_in:])
    if zc_after.size > 0:
        sit_rel = int(zc_after[0])
    else:
        # fallback: point where gyroX is closest to zero after x_peak_in
        sit_rel = int(np.argmin(np.abs(gyroX[x_peak_in:])))
    sit_ind = x_peak_in + sit_rel
    sit_ind = min(sit_ind, N - 1)
    events[sit_ind] = 7

    # ------------------------------------------------------------------
    # 7) Check that we at least have all labels present
    # ------------------------------------------------------------------
    uniq = np.unique(events).astype(int).tolist()
    # It’s possible that some events share the same sample → skip strict sequence check
    missing = {1, 2, 3, 4, 5, 6, 7} - set(uniq)
    if missing:
        raise RuntimeError(f"Not all TUG events detected. Missing: {sorted(missing)}")

    return events


# def detect_tug_events(acc: np.ndarray, gyr: np.ndarray) -> np.ndarray:
#     """
#     Detect and label events during a Timed Up and Go (TUG) test using IMU data.

#     This function identifies seven key events in the TUG test using gyroscope
#     and accelerometer signals from a lower-back mounted IMU sensor.

#     Parameters:
#     -----------
#     acc : np.ndarray
#         Accelerometer data with shape (3, n_samples) representing x, y, z axes
#     gyr : np.ndarray
#         Gyroscope data with shape (3, n_samples) representing x, y, z axes

#     Returns:
#     --------
#     events : np.ndarray
#         Array of same length as input signals with events labeled as:
#         0: No event
#         1: Sit to Stand start
#         2: Sit to Stand end / Walk forward start
#         3: Turn 1 start (Turn around cone)
#         4: Turn 1 end / Walk back start
#         5: Turn 2 start (Turn before sitting)
#         6: Turn 2 end / Stand to Sit start
#         7: Stand to Sit end

#     Notes:
#     ------
#     Event detection process:
#     1. Applies lowpass filter (1Hz cutoff) to reduce noise
#     2. Uses gyroscope X signal for sit-to-stand transitions (events 1,2)
#     3. Uses gyroscope Y signal squared for turn detection (events 3,4,5)
#     4. Uses peak detection and zero crossings to identify event boundaries
#     5. Sampling rate is assumed to be 100 Hz

#     The function asserts that all 7 events are detected in sequence.

#     Example:
#     --------
#     >>> events = detect_tug_events(accelerometer_data, gyroscope_data)
#     >>> event_times = np.where(events != 0)[0] / 100.0  # Convert to seconds
#     """
#     # Initialize lowpass filter (1 Hz cutoff, 100 Hz sampling)
#     sos = butter(10, 1, btype="lowpass", fs=100, output="sos")

#     # Extract individual axes
#     gyroX = gyr[0, :]  # Mainly captures anterior-posterior rotation
#     gyroY = gyr[1, :]  # Mainly captures mediolateral rotation
#     gyroZ = gyr[2, :]  # Mainly captures vertical rotation
#     accY = acc[1, :]  # Anterior-posterior acceleration

#     # Initialize events array
#     end = len(gyroX)
#     events = np.zeros((acc.shape[1],))

#     # Calculate squared gyroY signal for turn detection
#     pre_gyroY = np.power(gyroY, 2)
#     pre_gyroY_peaks = _get_highest_peak(pre_gyroY, th=0.4)

#     # Detect Sit-to-Stand initiation (Event 1)
#     min_ind = np.argmin(gyroX)
#     width_info = peak_widths(abs(gyroX), [min_ind], rel_height=0.9)
#     rising_ind = int(width_info[2])
#     events[rising_ind] = 1

#     # Detect Stand-to-Sit initiation (Event 6)
#     # Assume standing to sitting occurs after the last turn + 100ms to handle unrealistic angles
#     trunk_ind = pre_gyroY_peaks[-1] + 20
#     events[trunk_ind] = 6

#     # Detect final turn before sitting (Event 5)
#     zero_ind = _get_zero_crossing(gyroY)
#     fin_rot_ind = zero_ind[zero_ind < trunk_ind][-1]
#     events[fin_rot_ind] = 5

#     # # # Search for the sit-down flexion peak in gyroX after the final turn
#     # search_start = fin_rot_ind
#     # sit_flexion_peak = np.argmin(gyroX[search_start:]) + search_start

#     # # Find the onset of this flexion peak, similar to Event 1
#     # width_info_sit = peak_widths(abs(gyroX), [sit_flexion_peak], rel_height=0.9)
#     # stand_to_sit_start = int(width_info_sit[2])
#     # events[stand_to_sit_start] = 6

#     # Detect intermediate turn (Events 3,4)
#     pre_gyroY = sosfiltfilt(sos, np.power(gyroY[0:fin_rot_ind], 2))
#     int_rot_peak = _get_highest_peak(pre_gyroY)
#     rot_width_info = peak_widths(pre_gyroY, [int_rot_peak], rel_height=0.9)
#     left_rot = int(rot_width_info[2])
#     right_rot = int(rot_width_info[3])
#     events[left_rot] = 3  # Turn 1 start
#     events[right_rot] = 4  # Turn 1 end

#     # Detect Sit-to-Stand completion (Event 2)
#     sig = gyroX[rising_ind:left_rot]
#     rising_peak_idx = _get_highest_peak(sig)
#     rising_peak = _get_zero_crossing(sig[rising_peak_idx:])[0]
#     events[rising_ind + rising_peak + rising_peak_idx] = 2

#     # sig = gyroX[rising_ind:left_rot]
#     # rising_peak = _get_highest_peak(sig)
#     # events[rising_ind + rising_peak] = 2

#     # Detect final sitting (Event 7)
#     x_peaks = _get_highest_peak(gyroX[trunk_ind:end])
#     x_peak_in = x_peaks + trunk_ind
#     sit_ind = _get_zero_crossing(gyroX[x_peak_in:end])[0]
#     sit_ind = x_peak_in + sit_ind
#     events[sit_ind] = 7

#     # Verify all events were detected in sequence
#     assert np.unique(events).astype(int).tolist() == [0, 1, 2, 3, 4, 5, 6, 7], (
#         "Not all events detected"
#     )
#     return events


def calculate_tug_phase_durations(events):
    """
    Calculate temporal components (durations) between Timed Up and Go (TUG) test events.

    Parameters:
    -----------
    events : numpy.ndarray
        Array containing TUG event markers with the following encoding:
        1: Sit to Stand start
        2: Sit to Stand end / Walk forward start
        3: Turn 1 start (Turn around cone)
        4: Turn 1 end / Walk back start
        5: Turn 2 start (Turn before sitting)
        6: Turn 2 end / Stand to Sit start
        7: Stand to Sit end

    Returns:
    --------
    dict
        Dictionary containing durations (in seconds) for each TUG phase:
        - sit_to_stand: Duration of sit-to-stand transition
        - walk_forward: Duration of initial walking phase
        - turn_1: Duration of first 180° turn
        - walk_back: Duration of return walking phase
        - turn_2: Duration of second turn before sitting
        - stand_to_sit: Duration of stand-to-sit transition
        - total_duration: Total time to complete TUG test

    Notes:
    ------
    - Sampling rate is assumed to be 100 Hz (division by 100.0)
    - Uses np.argwhere to find timestamps of events
    - Durations are calculated by finding time differences between consecutive events

    Example:
    --------
    >>> events = np.array([0,1,0,0,2,0,3,0,4,0,5,0,6,0,7,0])
    >>> durations = calculate_tug_phase_durations(events)
    >>> print(f"Sit to stand duration: {durations['sit_to_stand'][0][0]:.2f} seconds")
    """
    phase_durations = {}

    # Calculate durations between consecutive events
    phase_durations["sit_to_stand [s]"] = (
        np.argwhere(events == 2) - np.argwhere(events == 1)
    ) / 100.0
    phase_durations["walk_forward [s]"] = (
        np.argwhere(events == 3) - np.argwhere(events == 2)
    ) / 100.0
    phase_durations["turn_1 [s]"] = (
        np.argwhere(events == 4) - np.argwhere(events == 3)
    ) / 100.0
    phase_durations["walk_back [s]"] = (
        np.argwhere(events == 5) - np.argwhere(events == 4)
    ) / 100.0
    phase_durations["turn_2 [s]"] = (
        np.argwhere(events == 6) - np.argwhere(events == 5)
    ) / 100.0
    phase_durations["stand_to_sit [s]"] = (
        np.argwhere(events == 7) - np.argwhere(events == 6)
    ) / 100.0

    # Calculate total TUG duration
    phase_durations["total_duration [s]"] = (
        np.argwhere(events == 7) - np.argwhere(events == 1)
    ) / 100.0

    return {k: v.item() for k, v in phase_durations.items()}


def estimate_orientation(acc_data, gyr_data, fs):
    """
    Estimates sensor orientation using the Madgwick filter with correct
    quaternion format handling between libraries.

    Parameters:
    -----------
    acc_data : np.ndarray
        3-axis accelerometer data (e.g., m/s^2).
    gyr_data : np.ndarray
        3-axis gyroscope data in rad/s.
    fs : float
        Sampling frequency in Hz.

    Returns:
    --------
    np.ndarray
        Euler angles (roll, pitch, yaw) in degrees for each sample.
    """
    num_samples = len(acc_data)
    madgwick = ahrs.filters.Madgwick(frequency=fs, gain=0.1)
    Q = np.zeros((num_samples, 4))

    # --- 1. Correct Initialization ---
    # Use a unit vector for the reference gravity
    g_e = np.array([0.0, 0.0, 1.0])
    a_s_initial = acc_data[0] / np.linalg.norm(acc_data[0])

    # Find the alignment rotation
    rot_align = Rotation.align_vectors(g_e[np.newaxis, :], a_s_initial[np.newaxis, :])[
        0
    ]

    # FIX: Use scalar_first=True to get [w, x, y, z] format for ahrs
    Q[0] = rot_align.inv().as_quat(scalar_first=True)
    # Q[0] = [1, 0, 0, 0]  # Temporary fix for initial orientation

    # --- 2. Process all subsequent samples ---
    # The Madgwick filter returns quaternions in [w, x, y, z] format
    for t in range(1, num_samples):
        Q[t] = madgwick.updateIMU(Q[t - 1], gyr_data[t], acc=acc_data[t])

    # --- 3. Correct Euler Conversion ---
    # FIX: Convert the [w, x, y, z] output from ahrs to the
    # [x, y, z, w] format required by scipy.
    Q_scipy = Q[:, [1, 2, 3, 0]]

    r = Rotation.from_quat(Q_scipy)
    euler_angles_deg = r.as_euler("zyx", degrees=True)
    # euler_angles_deg = r.as_euler("xyz", degrees=True)

    return euler_angles_deg


def calculate_kinematic_features(events, acc_data, gyr_data, fs=100.0):
    """
    Calculates kinematic parameters for different phases of the TUG test.
    (Corrected Version)
    """
    features = {}
    event_indices = {i: np.argwhere(events == i).item() for i in range(1, 8)}
    euler_angles = estimate_orientation(acc_data, gyr_data, fs)
    euler_angles[:, 2] = np.unwrap(euler_angles[:, 2], period=360)
    trunk_angle = euler_angles[:, 2] * -1  # Invert to simplify interpretation

    # 1. Sit to Stand (Event 1 to 2)
    start, end = event_indices[1], event_indices[2]
    acc_slice = acc_data[start:end]
    angle_slice = trunk_angle[start:end]

    # Plot trunk angle during Sit to Stand
    time_axis = np.arange(start, end) / fs
    # time_axis = np.arange(trunk_angle.shape[0]) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, angle_slice - angle_slice[0])
    # plt.plot(time_axis, trunk_angle)
    plt.title("Trunk Flexion/Extension Angle during Sit-to-Stand")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (°)")
    plt.grid(True)
    plt.show()

    features["Sit to stand_Antero-Posterior Acceleration range [m/s^2]"] = (
        acc_slice[:, 2].max() - acc_slice[:, 2].min()
    )
    features["Sit to stand_Lateral acceleration range [m/s^2]"] = (
        acc_slice[:, 0].max() - acc_slice[:, 0].min()
    )
    features["Sit to stand_Vertical acceleration range [m/s^2]"] = (
        acc_slice[:, 1].max() - acc_slice[:, 1].min()
    )
    # features["Sit to stand_Flexion peak [°]"] = angle_slice.max()
    # features["Sit to stand_Extension peak [°]"] = angle_slice.min()
    features["Sit to stand_Flexion range [°]"] = angle_slice.max() - angle_slice[0]

    features["Sit to stand_Extension range [°]"] = angle_slice.max() - angle_slice[-1]

    # 2. Stand to Sit (Event 6 to 7)
    start, end = event_indices[6], event_indices[7]
    acc_slice = acc_data[start:end]
    angle_slice = trunk_angle[start:end]

    # Plot trunk angle during Stand to Sit
    time_axis = np.arange(start, end) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, angle_slice - angle_slice[0])
    plt.title("Trunk Flexion/Extension Angle during Stand-to-Sit")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (°)")
    plt.grid(True)
    plt.show()

    features["Stand to sit_Antero-Posterior Acceleration range [m/s^2]"] = (
        acc_slice[:, 2].max() - acc_slice[:, 2].min()
    )
    features["Stand to sit_Lateral acceleration range [m/s^2]"] = (
        acc_slice[:, 0].max() - acc_slice[:, 0].min()
    )
    features["Stand to sit_Vertical acceleration range [m/s^2]"] = (
        acc_slice[:, 1].max() - acc_slice[:, 1].min()
    )
    # features["Stand to sit_Flexion peak [°]"] = angle_slice.max()
    # features["Stand to sit_Extension peak [°]"] = angle_slice.min()
    features["Stand to sit_Flexion range [°]"] = angle_slice.max() - np.min(
        angle_slice[: angle_slice.shape[0] // 2]
    )
    features["Stand to sit_Extension range [°]"] = angle_slice.max() - np.min(
        angle_slice[(angle_slice.shape[0] // 2) :]
    )

    # 3. Mid Turning (Turn 1, Event 3 to 4)
    start, end = event_indices[3], event_indices[4]
    gyr_slice = gyr_data[start:end] * (180.0 / np.pi)
    features["Mid turning_Peak angular speed [°/s]"] = np.abs(gyr_slice[:, 1]).max()
    features["Mid turning_Average angular speed [°/s]"] = np.abs(gyr_slice[:, 1]).mean()

    # 4. End Turning (Turn 2, Event 5 to 6)
    start, end = event_indices[5], event_indices[6]
    gyr_slice = gyr_data[start:end] * (180.0 / np.pi)
    features["End turning_Peak angular speed [°/s]"] = np.abs(gyr_slice[:, 1]).max()
    features["End turning_Average angular speed [°/s]"] = np.abs(gyr_slice[:, 1]).mean()

    return features
