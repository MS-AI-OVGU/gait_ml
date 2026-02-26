from typing import Any, Sequence, Tuple

import numpy as np
from scipy.ndimage import label
from scipy.stats import mode


class MetricsCalculator:
    """
    A class to calculate F1 score, accuracy, and related metrics.
    """

    def __init__(self, y_true: Sequence[Any], y_pred: Sequence[Any]) -> None:
        """
        Initializes the MetricsCalculator.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_f1_score(self) -> float:
        """
        Calculates the F1 score.

        Args:

        Returns:
            float: The F1 score.
        """
        TP, FP, FN, TN = self.calculate_confusion_matrix()

        if TP == 0:
            return 0.0  # or handle this case as you see fit (e.g., return None, raise an exception)
        f1_score = (2 * TP) / (
            2 * TP + FP + FN + 1e-7
        )  # Adding a small value to avoid division by zero

        return f1_score

    def calculate_accuracy(self) -> float:
        """
        Calculates the accuracy.

        Args:

        Returns:
            float: The accuracy.
        """
        TP, FP, FN, TN = self.calculate_confusion_matrix()
        accuracy = (TP + TN) / (TP + TN + FP + FN + +1e-7)
        return accuracy

    def calculate_confusion_matrix(self) -> Tuple[int, int, int, int]:
        """
        Calculates the true positives (TP), false positives (FP),
        false negatives (FN), and true negatives (TN).

        Args:

        Returns:
            tuple: A tuple containing TP, FP, FN, and TN.
        """
        TP = np.sum((self.y_true == 1) & (self.y_pred == 1))
        FP = np.sum((self.y_true == 0) & (self.y_pred == 1))
        FN = np.sum((self.y_true == 1) & (self.y_pred == 0))
        TN = np.sum((self.y_true == 0) & (self.y_pred == 0))

        return TP, FP, FN, TN

    def calculate_precision(self) -> float:
        """
        Calculates the precision.

        Args:

        Returns:
            float: The precision.
        """
        TP, FP, _, _ = self.calculate_confusion_matrix()
        if TP + FP == 0:
            return 0.0
        precision = TP / (TP + FP + 1e-7)
        return precision

    def calculate_recall(self) -> float:
        """
        Calculates the recall.

        Args:

        Returns:
            float: The recall.
        """
        TP, _, FN, _ = self.calculate_confusion_matrix()
        if TP + FN == 0:
            return 0.0
        recall = TP / (TP + FN + 1e-7)
        return recall


def merge_clustered_events(prediction: np.ndarray) -> np.ndarray:
    """
    Merges adjacent multiclass prediction events into single events.
    """
    # Automatically find clusters of adjacent non-zero elements
    labeled_array, num_clusters = label(prediction > 0)

    if num_clusters == 0:
        return prediction.copy()

    merged_prediction = np.zeros_like(prediction)

    # Process each cluster
    for i in range(1, num_clusters + 1):
        cluster_indices = np.where(labeled_array == i)[0]

        # Place the event at the middle of the cluster
        placement_idx = cluster_indices[len(cluster_indices) // 2]

        # Use the most frequent label (mode) in the cluster
        representative_label = mode(prediction[cluster_indices], keepdims=True).mode[0]
        merged_prediction[placement_idx] = representative_label

    return merged_prediction


def align_events(
    ground_truth: np.ndarray, prediction: np.ndarray, tolerance: int
) -> np.ndarray:
    """
    Aligns multiclass predicted events to the closest ground truth event
    of the same class within a tolerance window.
    """
    gt_indices = np.where(ground_truth > 0)[0]

    # Use a list of available prediction indices for easy removal
    available_preds = list(np.where(prediction > 0)[0])

    aligned_prediction = np.zeros_like(prediction)
    matches = {}  # Stores {gt_idx: matched_pred_idx}

    # Find the best match for each ground truth event
    for gt_idx in gt_indices:
        gt_label = ground_truth[gt_idx]
        best_dist = float("inf")
        best_match_idx = -1

        # Find the closest, available prediction of the same class
        for pred_idx in available_preds:
            if prediction[pred_idx] == gt_label:
                dist = abs(gt_idx - pred_idx)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_match_idx = pred_idx

        # If a match is found, lock it in and remove it from the pool
        if best_match_idx != -1:
            matches[gt_idx] = best_match_idx
            available_preds.remove(best_match_idx)

    # Build the final aligned array from the matches
    for gt_idx, pred_idx in matches.items():
        aligned_prediction[gt_idx] = prediction[pred_idx]  # True Positives

    # Add any remaining (unmatched) predictions as False Positives
    for pred_idx in available_preds:
        aligned_prediction[pred_idx] = prediction[pred_idx]

    return aligned_prediction


if __name__ == "__main__":
    # Example usage:
    true_labels = [1, 0, 1, 1, 0, 1]
    predicted_labels = [1, 1, 0, 1, 0, 0]

    calculator = MetricsCalculator(true_labels, predicted_labels)
    f1 = calculator.calculate_f1_score()
    accuracy = calculator.calculate_accuracy()
    tp, fp, fn, tn = calculator.calculate_confusion_matrix()

    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
