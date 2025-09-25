from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from scipy.signal import find_peaks, butter, sosfiltfilt


class GaitDataset(Dataset):
    def __init__(
        self,
        csv_files,  # Accepts a string or list of strings
        window_size=128,
        step_size=64,
        expand_labels=0,
        acc_sheet_name="Accelerometer",
        gyr_sheet_name="Gyroscope",
        zscale=True,
    ):
        """
        GaitDataset is a PyTorch Dataset class for loading and processing gait data from CSV files.

        Args:
            csv_files (str or list): Path(s) to the CSV file(s) containing IMU data.
            window_size (int): Length of each input sequence window.
            step_size (int): Step size for sliding window.
        """
        self.csv_files = csv_files
        self.window_size = window_size
        self.step_size = step_size
        self.expand_labels = expand_labels
        self.acc_sheet_name = acc_sheet_name
        self.gyr_sheet_name = gyr_sheet_name
        self.zscale = zscale

        # Load data from CSV files
        self.data = self._load_data()

    def _load_data(self):
        list_of_arrays = []
        self.raw_x = []
        self.raw_y = []
        self.filtered_x = []

        for file in self.csv_files:
            print(file)
            if file.endswith(".xls"):
                df_gyr = pd.read_excel(file, sheet_name=self.gyr_sheet_name)
                df_acc = pd.read_excel(file, sheet_name=self.acc_sheet_name)

                if df_gyr.shape[0] != df_acc.shape[0]:
                    df_acc = df_acc.iloc[: df_gyr.shape[0], :]
            else:
                raise ValueError(
                    "Unsupported file format. Please provide an Excel file."
                )

            arr_gyr = df_gyr.iloc[:, 1:].values
            arr_acc = df_acc.iloc[:, 1:].values
            raw_arr = np.concatenate([arr_gyr, arr_acc], axis=1)
            self.raw_x.append(raw_arr)
            # Filter data
            sos = butter(5, 5, btype="lowpass", fs=100, output="sos")
            filt1 = sosfiltfilt(sos, arr_gyr[:, 0])
            filt2 = sosfiltfilt(sos, arr_gyr[:, 1])
            filt3 = sosfiltfilt(sos, arr_gyr[:, 2])
            filtered_arr_gyr = np.vstack([filt1, filt2, filt3]).transpose()

            filt4 = sosfiltfilt(sos, arr_acc[:, 0])
            filt5 = sosfiltfilt(sos, arr_acc[:, 1])
            filt6 = sosfiltfilt(sos, arr_acc[:, 2])
            filtered_arr_acc = np.vstack([filt4, filt5, filt6]).transpose()
            filtered_arr = np.concatenate([filtered_arr_gyr, filtered_arr_acc], axis=1)
            self.filtered_x.append(filtered_arr)
            # Rule-based labelingarr_gyr
            # Detect heel strike (HS) and toe-off (TO) using CSAV and NZC method
            ms_peaks, _ = find_peaks(
                filtered_arr_gyr[:, 2], height=np.max(filtered_arr_gyr[:, 2] / 2)
            )
            hs_search_windows = np.diff(ms_peaks)
            detected_heel_strikes = []
            detected_toe_offs = []
            for cur_peak, cur_search_window in zip(ms_peaks, hs_search_windows):
                try:
                    cur_crop = arr_gyr[cur_peak : cur_peak + cur_search_window, 2]
                    cur_hs = find_peaks(-cur_crop, height=0)[0][0] + cur_peak
                    detected_heel_strikes.append(int(cur_hs))
                    ZP_idx = np.where(np.sign(cur_crop) < 0)[0][-1]
                    ZN_idx = np.where((cur_crop < 0))[0][0]
                    cur_to = cur_peak + ZN_idx + np.round((ZP_idx - ZN_idx) * 0.956)
                    # cur_to = ZP_idx
                    # cur_to += cur_peak
                    detected_toe_offs.append(cur_to.astype(int))
                except Exception as ERROR:
                    print(
                        f"Skipping: {cur_peak}:{cur_peak + cur_search_window} | {ERROR}"
                    )

            # Combine features and labels
            labels = np.zeros(arr_gyr.shape[0]).astype(int)
            labels[detected_heel_strikes] = 1
            labels[detected_toe_offs] = 2

            # Add labels before and after events
            if self.expand_labels > 0:
                for i in range(1, self.expand_labels + 1):
                    labels[list(np.array(detected_heel_strikes) + i)] = 1
                    labels[list(np.array(detected_heel_strikes) - i)] = 1
                    labels[list(np.array(detected_toe_offs) + i)] = 2
                    labels[list(np.array(detected_toe_offs) - i)] = 2

            self.raw_y.append(labels)

            final_array = np.concatenate(
                [filtered_arr, np.expand_dims(labels, 1)], axis=1
            )
            print("filtered_arr shape:", filtered_arr.shape)
            print("final_array shape:", final_array.shape)

            # Sliding window
            window_size = self.window_size
            stride = self.step_size
            num_dims = 7
            windows = np.lib.stride_tricks.sliding_window_view(
                final_array, window_shape=(window_size, num_dims)
            )

            cropped_arrays = windows[::stride, 0, :, :]
            print("cropped_array shape:", cropped_arrays.shape)
            list_of_arrays.append(cropped_arrays)

        # Concatenate all arrays from different files
        all_arrays = np.concatenate(list_of_arrays, axis=0)
        print("all_arrays shape:", all_arrays.shape)

        # Standardize the first three channels (x, y, z) of the IMU data and keep the labels intact
        if self.zscale:
            zscaled_data = (
                all_arrays[:, :, :3] - all_arrays[:, :, :3].mean((0, 1))
            ) / all_arrays[:, :, :3].std((0, 1))
            all_arrays = np.concatenate([zscaled_data, all_arrays[:, :, 3:]], axis=-1)
        return all_arrays

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        input_seq = self.data[idx, :, :-1]
        labels = self.data[idx, :, -1].astype(int)  # shape: (window_size,)
        # One-hot encode labels
        num_classes = int(self.data[:, :, -1].max()) + 1  # assumes labels are 0-indexed
        labels_onehot = torch.nn.functional.one_hot(
            torch.tensor(labels), num_classes=num_classes
        ).float()
        return torch.tensor(input_seq, dtype=torch.float32), labels_onehot


if __name__ == "__main__":
    # Example usage for GaitDataset
    xls_files = glob("/home/qivy00li/projects/spine_interaction/data/raw/train/*.xls")
    window_size = 128
    step_size = 64
    batch_size = 8

    dataset = GaitDataset(
        csv_files=xls_files,
        window_size=window_size,
        step_size=step_size,
    )
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        inputs, labels_onehot = batch
        print("Inputs shape:", inputs.shape)
        print("Labels (one-hot) shape:", labels_onehot.shape)
        break  # Just to show one batch
