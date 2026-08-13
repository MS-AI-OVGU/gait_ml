import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_gait_phases_vectorized(
    label_array: np.ndarray, gait_cycle_limit: float = 300.0
):
    """
    Calculates the stance and swing time as a percentage of the gait cycle
    using vectorized NumPy operations. (Robust to 0D array inputs)

    Args:
        label_array (np.ndarray): A 1D array or scalar where:
                                  0 = Not Event (NE)
                                  1 = Initial Contact (IC)
                                  2 = Foot Off (FO)

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                                       1. stance_percentages (one value per cycle)
                                       2. swing_percentages (one value per cycle)
    """

    # --- FIX: Ensure input is at least a 1D array ---
    # This prevents the "Calling nonzero on 0d arrays" error
    # if the input is a single scalar value.
    label_array = np.atleast_1d(label_array)

    # 1. Find the indices (frames) of all IC and FO events
    ic_indices = np.where(label_array == 1)[0]
    fo_indices = np.where(label_array == 2)[0]
    # print(len(ic_indices))
    # print(len(fo_indices))
    # --- Early exit for insufficient data ---
    if len(ic_indices) < 2 or len(fo_indices) == 0:
        print("Not sufficient labels!")
        return np.array([]), np.array([])

    # 2. Define all potential cycles
    start_ics = ic_indices[:-1]
    end_ics = ic_indices[1:]

    # 3. Find the matching FO for each cycle
    fo_target_indices = np.searchsorted(fo_indices, start_ics)

    # --- Filter 1: Handle boundary conditions ---
    valid_target_mask = fo_target_indices < len(fo_indices)

    # if not np.any(valid_target_mask):
    #     return np.array([]), np.array([])

    start_ics = start_ics[valid_target_mask]
    end_ics = end_ics[valid_target_mask]
    fo_target_indices = fo_target_indices[valid_target_mask]

    mid_fos = fo_indices[fo_target_indices]

    # --- Filter 2: Ensure data is clean (IC < FO < next_IC) ---
    valid_cycle_mask = (mid_fos > start_ics) & (mid_fos < end_ics)

    if not np.any(valid_cycle_mask):
        return np.array([]), np.array([])

    final_start_ics = start_ics[valid_cycle_mask]
    final_end_ics = end_ics[valid_cycle_mask]
    final_mid_fos = mid_fos[valid_cycle_mask]

    # 4. Calculate all durations
    cycle_durations = final_end_ics - final_start_ics
    stance_durations = final_mid_fos - final_start_ics
    # swing_durations = final_end_ics - final_mid_fos

    # 5. Calculate percentages
    epsilon = 1e-7
    stance_percentages = (stance_durations / (cycle_durations + epsilon)) * 100

    # Will only return valid gait cycles (currentduration 3secs)
    stance_percentages = stance_percentages[cycle_durations <= gait_cycle_limit]
    return stance_percentages


def aggregate_res(stance_pred, subject_ids):
    combined_res = {}
    for k, v in zip(subject_ids, stance_pred):
        if k in combined_res:
            combined_res[k] = np.concatenate((combined_res[k], v))
        else:
            combined_res[k] = v

    combined_res = pd.Series({k: v.mean().item() for k, v in combined_res.items()})
    return pd.DataFrame(combined_res)


def lins_ccc(x, y):
    """
    Calculate Lin's Concordance Correlation Coefficient.
    (No changes from previous script)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)

    cov_matrix = np.cov(x, y, ddof=1)
    cov_xy = cov_matrix[0, 1]

    numerator = 2 * cov_xy
    denominator = var_x + var_y + (mean_x - mean_y) ** 2

    if denominator == 0:
        return 1.0 if np.all(x == y) else 0.0

    return numerator / denominator


def plot_bland_altman_publicationv2(
    method1,
    method2,
    method1_name="LE-GRU",
    method2_name="Reference",
    units="ms",
    display_scale=1000.0,  # Convert input seconds to milliseconds
    ax=None,
    filename="bland_altman_plot.pdf",
    feature_name="Stride time",
    xpos=0.95,
    ypos=0.90,
):
    """
    Create a publication-ready Bland–Altman plot.

    Parameters
    ----------
    method1, method2 : array-like
        Paired measurements. Values are assumed to be in seconds when
        display_scale=1000.
    units : str
        Display unit, e.g. "ms".
    display_scale : float
        Conversion from the input unit to the display unit.
        Use 1000 for seconds -> milliseconds and 1 for no conversion.
    """

    font_sizes = {
        "main": 18,
        "axes_label": 25,
        "tick_label": 25,
        "metrics_box": 20,
        "line_label": 20,
    }

    plt.style.use("seaborn-v0_8-white")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_sizes["main"],
            "axes.labelsize": font_sizes["axes_label"],
            "xtick.labelsize": font_sizes["tick_label"],
            "ytick.labelsize": font_sizes["tick_label"],
            "axes.linewidth": 1.5,
        }
    )

    # Convert and validate paired inputs
    method1 = np.asarray(method1, dtype=float).ravel()
    method2 = np.asarray(method2, dtype=float).ravel()

    if method1.shape != method2.shape:
        raise ValueError("method1 and method2 must have the same shape.")

    valid = np.isfinite(method1) & np.isfinite(method2)
    method1 = method1[valid]
    method2 = method2[valid]

    if method1.size < 2:
        raise ValueError("At least two valid paired observations are required.")

    # Bland–Altman values in the original input unit
    average = (method1 + method2) / 2.0
    difference = method1 - method2

    mean_diff = np.mean(difference)
    std_diff = np.std(difference, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    mae = mean_absolute_error(method2, method1)
    rmse = np.sqrt(mean_squared_error(method2, method1))
    # ccc = lins_ccc(method2, method1)
    ccc, ccc_ci_lower, ccc_ci_upper = lins_ccc_with_ci(
        method2,
        method1,
        confidence_level=0.95,
        n_resamples=10_000,
        random_seed=42,
    )

    # Convert values for display
    average_display = average * display_scale
    difference_display = difference * display_scale
    mean_diff_display = mean_diff * display_scale
    loa_upper_display = loa_upper * display_scale
    loa_lower_display = loa_lower * display_scale
    mae_display = mae * display_scale
    rmse_display = rmse * display_scale

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.scatter(
        average_display,
        difference_display,
        facecolors="none",
        edgecolors="black",
        s=60,
        alpha=0.6,
    )

    ax.axhline(
        mean_diff_display,
        color="black",
        linestyle="--",
        linewidth=2,
    )
    ax.axhline(
        loa_upper_display,
        color="dimgray",
        linestyle=":",
        linewidth=2,
    )
    ax.axhline(
        loa_lower_display,
        color="dimgray",
        linestyle=":",
        linewidth=2,
    )

    # metrics_text = (
    #     f"RMSE: {rmse_display:.1f} {units}\n"
    #     f"MAE: {mae_display:.1f} {units}\n"
    #     f"Lin's CCC: {ccc:.3f}"
    # )
    metrics_text = (
        f"RMSE: {rmse_display:.1f} {units}\n"
        f"MAE: {mae_display:.1f} {units}\n"
        f"Lin's CCC: {ccc:.3f}\n"
        f"95% CI: [{ccc_ci_lower:.3f}, {ccc_ci_upper:.3f}]"
    )

    ax.text(
        xpos,
        ypos,
        metrics_text,
        transform=ax.transAxes,
        fontsize=font_sizes["metrics_box"],
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.8,
            linewidth=1,
        ),
    )

    ax.set_xlabel(
        f"Mean {feature_name} [{units}]",
        fontsize=font_sizes["axes_label"],
        fontweight="bold",
    )
    ax.set_ylabel(
        rf"$\Delta$ {feature_name} [{units}]",
        fontsize=font_sizes["axes_label"],
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=1.5,
        length=6,
    )

    # Ensure axis limits are finalized before positioning line labels
    ax.relim()
    ax.autoscale_view()

    x_min, x_max = ax.get_xlim()
    x_padding = 0.01 * (x_max - x_min)
    x_text = x_max - x_padding

    label_box = dict(
        facecolor="white",
        alpha=0.8,
        edgecolor="none",
        pad=0.1,
    )

    ax.text(
        x_text,
        mean_diff_display,
        f"Bias: {mean_diff_display:.1f} {units}",
        ha="right",
        va="bottom",
        fontsize=font_sizes["line_label"],
        color="black",
        bbox=label_box,
    )

    ax.text(
        x_text,
        loa_upper_display,
        f"Upper 95% LoA: {loa_upper_display:.1f} {units}",
        ha="right",
        va="bottom",
        fontsize=font_sizes["line_label"],
        color="dimgray",
        bbox=label_box,
    )

    ax.text(
        x_text,
        loa_lower_display,
        f"Lower 95% LoA: {loa_lower_display:.1f} {units}",
        ha="right",
        va="top",
        fontsize=font_sizes["line_label"],
        color="dimgray",
        bbox=label_box,
    )

    fig.tight_layout()
    fig.savefig(
        filename,
        dpi=600,
        bbox_inches="tight",
        transparent=False,
    )

    return fig, ax


from scipy.stats import bootstrap


def lins_ccc_with_ci(
    method1,
    method2,
    confidence_level=0.95,
    n_resamples=10_000,
    random_seed=42,
):
    """
    Calculate Lin's CCC with a paired BCa bootstrap confidence interval.

    Each row must represent one independent observational unit,
    typically one participant.
    """

    method1 = np.asarray(method1, dtype=float).ravel()
    method2 = np.asarray(method2, dtype=float).ravel()

    if method1.shape != method2.shape:
        raise ValueError("method1 and method2 must have the same shape.")

    valid = np.isfinite(method1) & np.isfinite(method2)
    method1 = method1[valid]
    method2 = method2[valid]

    if method1.size < 3:
        raise ValueError("At least three valid paired observations are required.")

    def ccc_statistic(x, y):
        return lins_ccc(x, y)

    ccc = ccc_statistic(method1, method2)

    result = bootstrap(
        data=(method1, method2),
        statistic=ccc_statistic,
        paired=True,
        vectorized=False,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method="BCa",
        rng=np.random.default_rng(random_seed),
    )

    ci_lower = float(result.confidence_interval.low)
    ci_upper = float(result.confidence_interval.high)

    return float(ccc), ci_lower, ci_upper


def plot_bland_altman_publication(
    method1,
    method2,
    method1_name="IMU",
    method2_name="VICON",
    units="°",
    ax=None,
    filename="bland_altman_plot.pdf",
    feature_name=None,
    xpos=0.95,
    ypos=0.9,
):
    """
    Creates a publication-ready Bland-Altman plot with easily modifiable fonts.
    """

    # --- MODIFICATION: Centralized Font Size Control ---
    # Easily modify all font sizes here
    font_sizes = {
        "main": 18,  # Base font size
        "axes_label": 25,  # X and Y axis labels
        "tick_label": 25,  # X and Y tick numbers
        "metrics_box": 20,  # RMSE, MAE, CCC box
        "line_label": 20,  # Bias and LoA line labels
    }
    # --- END MODIFICATION ---

    # --- 1. Set Plot Style ---
    plt.style.use("seaborn-v0_8-white")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_sizes["main"],  # Use base size
            "axes.labelsize": font_sizes["axes_label"],  # Use axes label size
            "xtick.labelsize": font_sizes["tick_label"],  # Use tick label size
            "ytick.labelsize": font_sizes["tick_label"],  # Use tick label size
            "axes.linewidth": 1.5,
        }
    )

    # --- 2. Calculate Data ---
    method1 = np.asarray(method1)
    method2 = np.asarray(method2)

    average = (method1 + method2) / 2
    difference = method1 - method2

    # --- 3. Calculate Statistics ---
    mean_diff = np.mean(difference)
    std_diff = np.std(difference, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    mae = mean_absolute_error(method2, method1)
    rmse = np.sqrt(mean_squared_error(method2, method1))
    ccc = lins_ccc(method2, method1)

    # --- 4. Create the Plot ---
    if ax is None:
        # fig, ax = plt.subplots(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    ax.scatter(
        average,
        difference,
        c="black",
        facecolors="none",
        edgecolors="black",
        s=60,
        alpha=0.6,
        label="Data Points",
    )

    # Statistical lines
    ax.axhline(mean_diff, color="black", linestyle="--", linewidth=2)
    ax.axhline(loa_upper, color="dimgray", linestyle=":", linewidth=2)
    ax.axhline(loa_lower, color="dimgray", linestyle=":", linewidth=2)

    # --- 5. Add Metrics Text Box ---
    text_str = f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nLin's CCC: {ccc:.3f}"
    props = dict(
        boxstyle="round", facecolor="white", edgecolor="black", alpha=0.6, lw=1
    )
    ax.text(
        xpos,
        ypos,
        text_str,
        transform=ax.transAxes,
        fontsize=font_sizes["metrics_box"],  # Use metrics box size
        # fontweight='bold',
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    # --- 6. Final Touches ---

    # Use axes label size
    ax.set_xlabel(
        # f"Average of {method1_name} and {method2_name} ({units})",
        f"Mean {feature_name}" + f"{units}",
        fontsize=font_sizes["axes_label"],
        fontweight="bold",
    )
    ax.set_ylabel(
        # f"Difference (New-Reference){method1_name} and {method2_name} ({units})",
        r"$\Delta$ " + f"{feature_name}" + f"{units}",
        fontsize=font_sizes["axes_label"],
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", which="major", direction="out", width=1.5, length=6)

    # Add text labels directly to lines
    x_pos = ax.get_xlim()[1]
    x_pos_with_padding = x_pos - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01

    # Use line label size
    ax.text(
        x_pos_with_padding,
        mean_diff,
        f" Mean (Bias): {mean_diff:.2f} {units}",
        ha="right",
        va="bottom",
        fontsize=font_sizes["line_label"],
        color="black",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.1),
    )

    # Use line label size
    ax.text(
        x_pos_with_padding,
        loa_upper,
        f" +1.96 SD: {loa_upper:.2f} {units}",
        ha="right",
        va="bottom",
        fontsize=font_sizes["line_label"],
        color="dimgray",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.1),
    )

    # Use line label size
    ax.text(
        x_pos_with_padding,
        loa_lower,
        f" -1.96 SD: {loa_lower:.2f} {units}",
        ha="right",
        va="top",
        fontsize=font_sizes["line_label"],
        color="dimgray",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.1),
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight", transparent=False)

    return fig, ax


def plot_confusion_matrix(
    cm,
    class_names,
    title=None,
    fmt="d",
    cbar_label="Count",
    cmap=plt.cm.Blues,
    file_name="confusion_matrix.pdf",
):
    """
    Plots a publication-ready confusion matrix for JNER.
    """

    # 1. Set JNER-compliant font (Arial/Helvetica is standard)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # 2. Size: 3.5 inches is standard for single-column width (85-90mm)
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=1.0,  # Thicker lines for better separation in print
        linecolor="black",
        cbar=False,  # Disable colorbar if numbers are annotated (saves space)
        annot_kws={
            "fontsize": 18,
            "fontweight": "bold",
        },  # Large font for readability when resized
        ax=ax,
        square=True,
    )

    # 3. Titles: JNER prefers titles in the caption, not the image.
    # Only set if strictly necessary for internal use.
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # 4. Axis Labels: Clear and large
    ax.set_ylabel("True Event", fontsize=18, fontweight="bold")
    ax.set_xlabel("Predicted Event", fontsize=18, fontweight="bold")

    # 5. Ticks: Center them and ensure readability
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks + 0.5)
    ax.set_yticks(tick_marks + 0.5)

    ax.set_xticklabels(class_names, fontsize=16, fontweight="medium")
    # CHANGED: Rotation 0 is better for short labels like "IC/FO"
    ax.set_yticklabels(
        class_names, fontsize=16, fontweight="medium", rotation=0, va="center"
    )

    # Cleanups
    ax.tick_params(axis="both", which="major", length=0)
    plt.tight_layout()

    # 6. Saving: Use 600 DPI for raster or PDF/EPS for vector (Best for JNER)
    # If saving as PNG, use 600 dpi. If PDF, dpi is less critical but good practice.
    plt.savefig(file_name, dpi=600, bbox_inches="tight", transparent=False)

    # plt.close(fig) # Uncomment to prevent display in notebooks if generating many
    return fig


def align_imu_to_acc(t_acc, ax, ay, az, t_gyr, wx, wy, wz):
    """Align gyroscope data to accelerometer timestamps using linear interpolation."""
    # 1) sort by time (np.interp expects increasing x)
    ia = np.argsort(t_acc)
    t_acc, ax, ay, az = t_acc[ia], ax[ia], ay[ia], az[ia]
    ig = np.argsort(t_gyr)
    t_gyr, wx, wy, wz = t_gyr[ig], wx[ig], wy[ig], wz[ig]

    # 2) start both at 0
    t_acc = t_acc - t_acc[0]
    t_gyr = t_gyr - t_gyr[0]

    # 3) use only overlapping time range (avoid extrapolation)
    t0, t1 = max(t_acc[0], t_gyr[0]), min(t_acc[-1], t_gyr[-1])
    m = (t_acc >= t0) & (t_acc <= t1)
    t = t_acc[m]

    # 4) interpolate gyro onto accel timestamps
    wx_i = np.interp(t, t_gyr, wx)
    wy_i = np.interp(t, t_gyr, wy)
    wz_i = np.interp(t, t_gyr, wz)

    return t, ax[m], ay[m], az[m], wx_i, wy_i, wz_i


def calculate_stride_times(
    label_array: np.ndarray,
    sampling_rate_hz: float,
    gait_cycle_limit: float = 300.0,
) -> np.ndarray:
    """Calculate stride times in seconds from consecutive IC events."""

    label_array = np.atleast_1d(label_array)

    # Find consecutive IC regions
    ic_mask = label_array == 1
    ic_starts = np.flatnonzero(ic_mask & ~np.r_[False, ic_mask[:-1]])
    ic_ends = np.flatnonzero(ic_mask & ~np.r_[ic_mask[1:], False])

    # Use the middle sample of each IC prediction
    ic_indices = (ic_starts + ic_ends) // 2

    if len(ic_indices) < 2:
        return np.array([])

    # Consecutive IC-to-IC durations
    stride_durations = np.diff(ic_indices)

    # Remove implausibly long cycles
    stride_durations = stride_durations[
        (stride_durations > 0) & (stride_durations <= gait_cycle_limit)
    ]

    return stride_durations / sampling_rate_hz
