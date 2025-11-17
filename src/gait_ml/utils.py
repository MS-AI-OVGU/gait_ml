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


def plot_bland_altman_publication(
    method1, method2, method1_name="IMU", method2_name="VICON", units="°", ax=None
):
    """
    Creates a publication-ready Bland-Altman plot with easily modifiable fonts.
    """

    # --- MODIFICATION: Centralized Font Size Control ---
    # Easily modify all font sizes here
    font_sizes = {
        "main": 18,  # Base font size
        "axes_label": 18,  # X and Y axis labels
        "tick_label": 18,  # X and Y tick numbers
        "metrics_box": 18,  # RMSE, MAE, CCC box
        "line_label": 18,  # Bias and LoA line labels
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
        boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9, lw=1
    )
    ax.text(
        0.95,
        0.90,
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
        f"Average of {method1_name} and {method2_name} ({units})",
        fontsize=font_sizes["axes_label"],
        fontweight="bold",
    )
    ax.set_ylabel(
        f"Difference ({method1_name} - {method2_name}) ({units})",
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

    return fig, ax


def plot_confusion_matrix(
    cm,
    class_names,
    title,
    fmt,
    cbar_label,
    cmap=plt.cm.Blues,
    file_name="confusion_matrix.png",
    normalize=None,
):
    """
    Plots a confusion matrix heatmap.

    Args:
        cm (np.array): The confusion matrix (absolute counts).
        class_names (list): List of class names for labels.
        title (str): Title of the plot.
        fmt (str): Format string for annotations (e.g., 'd' for counts, '.2f' for percentages).
        cbar_label (str): Label for the color bar (if cbar is True).
        cmap (plt.cm): Matplotlib colormap.
        file_name (str): Name to save the figure.
        normalize (str, optional): Normalization type.
                                   'true' (row-wise, Recall),
                                   'pred' (column-wise, Precision),
                                   'all' (total counts).
                                   Defaults to None (absolute counts).
    """

    # --- 1. Normalization Logic ---
    if normalize == "true":
        # Normalize by True Label counts (Rows sum to 1.0)
        # Sum along axis 1 (rows) and tile to match the shape of cm
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    elif normalize == "pred":
        # Normalize by Predicted Label counts (Columns sum to 1.0)
        # Sum along axis 0 (columns)
        cm = cm.astype("float") / cm.sum(axis=0)
    elif normalize == "all":
        # Normalize by the grand total
        cm = cm.astype("float") / cm.sum()

    # Handle NaN values that result from dividing by zero (empty classes)
    cm = np.nan_to_num(cm)

    # --- 2. Plotting ---
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor="black",
        cbar=False,
        # cbar_kws={'label': cbar_label, 'orientation': 'vertical', 'pad': 0.04, 'aspect': 30},
        annot_kws={"fontsize": 16, "fontweight": "bold"},
        ax=ax,
        square=True,
    )

    # --- 3. Labels and Ticks (Unchanged) ---
    ax.set_title(title, fontsize=12, fontweight="bold", pad=5)
    ax.set_ylabel("True Label", fontsize=20, fontweight="medium")
    ax.set_xlabel("Predicted Label", fontsize=20, fontweight="medium")

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks + 0.5)
    ax.set_yticks(tick_marks + 0.5)
    ax.set_xticklabels(class_names, fontsize=16)
    ax.set_yticklabels(class_names, fontsize=16, rotation=90, va="center")

    ax.set_ylim(len(class_names), 0)
    ax.tick_params(axis="both", which="major", length=0)

    plt.tight_layout()

    # fig.show()
    # plt.close(fig)
    return fig
