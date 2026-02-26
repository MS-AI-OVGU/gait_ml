from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import bisect
from scipy.spatial.transform import Rotation
from gaitmap.utils.rotations import rotate_dataset
from gaitmap.preprocessing import sensor_alignment
from scipy.signal import find_peaks, butter, sosfiltfilt

from gait_ml import utils


class GaitDataset(Dataset):
    def __init__(
        self,
        csv_files,  # Accepts a string or list of strings
        window_size=128,
        step_size=64,
        expand_labels=0,
        acc_sheet_name="Accelerometer",
        gyr_sheet_name="Gyroscope",
        zscale=None,
        zscale_stats=None,
    ):
        """
        GaitDataset is a PyTorch Dataset class for loading and processing gait data from CSV files.
        """
        self.csv_files = csv_files
        self.window_size = window_size
        self.step_size = step_size
        self.expand_labels = expand_labels
        self.acc_sheet_name = acc_sheet_name
        self.gyr_sheet_name = gyr_sheet_name
        self.zscale = zscale
        self.zscale_stats = zscale_stats

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
                    df_aligned = utils.align_imu_to_acc(
                        df_acc.values[:, 0],
                        df_acc.values[:, 1],
                        df_acc.values[:, 2],
                        df_acc.values[:, 3],
                        df_gyr.values[:, 0],
                        df_gyr.values[:, 1],
                        df_gyr.values[:, 2],
                        df_gyr.values[:, 3],
                    )
                    df_acc = pd.DataFrame(np.stack(df_aligned[:4]).T)
                    df_gyr = pd.DataFrame(
                        np.stack(
                            [df_aligned[0], df_aligned[4], df_aligned[5], df_aligned[6]]
                        ).T
                    )
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
            for idx in np.arange(len(ms_peaks)):
                cur_peak = ms_peaks[idx]
                if idx == len(ms_peaks) - 1:
                    cur_search_window = hs_search_windows[idx - 1]
                else:
                    cur_search_window = hs_search_windows[idx]
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

                    # Make sure the last event is HeelStrike
                    if idx == len(ms_peaks) - 1:
                        if detected_toe_offs[-1] > detected_heel_strikes[-1]:
                            detected_toe_offs = detected_toe_offs[:-1]
                except Exception as ERROR:
                    print(
                        f"Skipping: {cur_peak}:{cur_peak + cur_search_window} | {ERROR}"
                    )
            ####################################################################################
            #### Manual correction of labels
            ####################################################################################
            if "10_1_2mW_IPhone.xls" in file:
                # Remove the first detected toe-off and heel-strike
                detected_toe_offs = detected_toe_offs[1:]
                detected_heel_strikes = detected_heel_strikes[1:]

            if "18_1_2mW_IPhone.xls" in file:
                item_idx = detected_toe_offs.index(9732)
                detected_toe_offs[item_idx] = detected_toe_offs[item_idx] - 10

            if "25_1_2mW_IPhone.xls" in file:
                for idx_ in [4471, 7566]:
                    curr_item_idx = detected_toe_offs.index(idx_)
                    detected_toe_offs[curr_item_idx] = (
                        detected_toe_offs[curr_item_idx] - 10
                    )
            if "28_1_2mW_IPhone.xls" in file:
                item_idx = detected_toe_offs.index(9927)
                detected_toe_offs[item_idx] = detected_toe_offs[item_idx] - 10

            if "32_1_2mW_IPhone.xls" in file:
                item_idx = detected_toe_offs.index(11269)
                detected_toe_offs[item_idx] = detected_toe_offs[item_idx] - 10

            if "33_1_2mW_IPhone.xls" in file:
                for idx_ in [2640, 3581]:
                    bisect.insort(detected_heel_strikes, idx_)

                for idx_ in [2597, 3535]:
                    bisect.insort(detected_toe_offs, idx_)

            ####################################################################################
            #### End of manual correction of labels
            ####################################################################################

            # Combine features and labels
            labels = np.zeros(arr_gyr.shape[0]).astype(int)
            labels[detected_heel_strikes] = 1
            labels[detected_toe_offs] = 2

            # Add labels before and after events
            if self.expand_labels > 0:
                for i in range(1, self.expand_labels + 1):
                    # labels[list(np.array(detected_heel_strikes) + i)] = 1
                    # labels[list(np.array(detected_heel_strikes) - i)] = 1
                    # labels[list(np.array(detected_toe_offs) + i)] = 2
                    # labels[list(np.array(detected_toe_offs) - i)] = 2
                    # Ensure indices are within bounds for heel strikes
                    valid_indices_plus = np.array(detected_heel_strikes) + i
                    valid_indices_minus = np.array(detected_heel_strikes) - i
                    valid_indices_plus = valid_indices_plus[
                        valid_indices_plus < len(labels)
                    ]
                    valid_indices_minus = valid_indices_minus[valid_indices_minus >= 0]
                    labels[valid_indices_plus] = 1
                    labels[valid_indices_minus] = 1

                    # Ensure indices are within bounds for toe offs
                    valid_indices_plus = np.array(detected_toe_offs) + i
                    valid_indices_minus = np.array(detected_toe_offs) - i
                    valid_indices_plus = valid_indices_plus[
                        valid_indices_plus < len(labels)
                    ]
                    valid_indices_minus = valid_indices_minus[valid_indices_minus >= 0]
                    labels[valid_indices_plus] = 2
                    labels[valid_indices_minus] = 2
            self.raw_y.append(labels)

            final_array = np.concatenate(
                [filtered_arr, np.expand_dims(labels, 1)], axis=1
            )
            # print("filtered_arr shape:", filtered_arr.shape)
            # print("final_array shape:", final_array.shape)

            # list_of_arrays.append(final_array)
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

        # # Concatenate all arrays from different files
        all_arrays = np.concatenate(list_of_arrays, axis=0)
        print("all_arrays shape:", all_arrays.shape)

        # Standardize the first three channels (x, y, z) of the IMU data and keep the labels intact
        R = Rotation.from_matrix(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
        if self.zscale:
            temp_array = all_arrays[:, :, :-1]
            # Align data to global coordinate system
            data_sf = [
                pd.DataFrame(
                    temp_array[i],
                    columns=["gyr_x", "gyr_y", "gyr_z", "acc_x", "acc_y", "acc_z"],
                )
                for i in range(temp_array.shape[0])
            ]
            data_sf = [rotate_dataset(i, R) for i in data_sf]
            data_sf = np.stack(([i.values for i in data_sf]))

            if isinstance(self.zscale, dict):
                curr_mean = np.asarray(self.zscale.get("mean"))
                curr_std = np.asarray(self.zscale.get("std"))
            else:
                curr_mean = data_sf.mean((0, 1))
                curr_std = data_sf.std((0, 1))
            zscaled_data = (data_sf - curr_mean) / curr_std
            self.zscale_stats = {"mean": curr_mean, "std": curr_std}
            all_arrays = np.concatenate([zscaled_data, all_arrays[:, :, -1:]], axis=-1)
        return all_arrays

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # cropped_arrays = windows[::stride, 0, :, :]
        input_features = self.data[idx, :, :-1]
        output_labels = self.data[idx, :, -1]
        return (
            torch.tensor(input_features, dtype=torch.float32),
            torch.tensor(output_labels, dtype=torch.long),
        )


class ExternalGaitDataset1(Dataset):
    def __init__(
        self,
        csv_files,  # Accepts a string or list of strings
        window_size=128,
        step_size=64,
        expand_labels=0,
        acc_sheet_name="Accelerometer",
        gyr_sheet_name="Gyroscope",
        zscale=None,
    ):
        """
        GaitDataset is a PyTorch Dataset class for loading and processing gait data from CSV files.
        """
        super().__init__()
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
            if file.endswith(".npy"):
                loaded_data = np.load(file, allow_pickle=True)
                arr_gyr = loaded_data[:, [1, 2, 3]]
                arr_acc = loaded_data[:, [4, 5, 6]]
            else:
                raise ValueError(
                    "Unsupported file format. Please provide an Excel file."
                )

            # arr_gyr = df_gyr.iloc[:, 1:].values
            # arr_acc = df_acc.iloc[:, 1:].values
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

            # Combine features and labels
            labels = loaded_data[:, -1]
            # labels[detected_heel_strikes] = 1
            # labels[detected_toe_offs] = 2

            # Add labels before and after events
            print("self.expand_labels", self.expand_labels)
            if self.expand_labels > 0:
                print("=======================================")
                print("=== Updating labels with expansion...")
                print("=======================================")
                for i in range(1, self.expand_labels + 1):
                    # Ensure indices are within bounds for heel strikes
                    detected_heel_strikes = np.where(labels == 1)[0]
                    detected_toe_offs = np.where(labels == 2)[0]
                    valid_indices_plus = detected_heel_strikes + i
                    valid_indices_minus = detected_heel_strikes - i
                    valid_indices_plus = valid_indices_plus[
                        valid_indices_plus < len(labels)
                    ]
                    valid_indices_minus = valid_indices_minus[valid_indices_minus >= 0]
                    labels[valid_indices_plus] = 1
                    labels[valid_indices_minus] = 1

                    # Ensure indices are within bounds for toe offs
                    valid_indices_plus = detected_toe_offs + i
                    valid_indices_minus = detected_toe_offs - i
                    valid_indices_plus = valid_indices_plus[
                        valid_indices_plus < len(labels)
                    ]
                    valid_indices_minus = valid_indices_minus[valid_indices_minus >= 0]
                    labels[valid_indices_plus] = 2
                    labels[valid_indices_minus] = 2
            self.raw_y.append(labels)

            final_array = np.concatenate(
                [filtered_arr, np.expand_dims(labels, 1)], axis=1
            )
            # print("filtered_arr shape:", filtered_arr.shape)
            # print("final_array shape:", final_array.shape)

            # list_of_arrays.append(final_array)
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

        # # Concatenate all arrays from different files
        all_arrays = np.concatenate(list_of_arrays, axis=0)
        print("all_arrays shape:", all_arrays.shape)

        # Standardize the first three channels (x, y, z) of the IMU data and keep the labels intact
        # Standardize the first three channels (x, y, z) of the IMU data and keep the labels intact
        R = Rotation.from_matrix(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
        if self.zscale:
            print("====++++Enters ZSCALE in ExternalGAITDATASET====++++")
            temp_array = all_arrays[:, :, :-1]
            # Align data to global coordinate system
            data_sf = [
                pd.DataFrame(
                    temp_array[i],
                    columns=["gyr_x", "gyr_y", "gyr_z", "acc_x", "acc_y", "acc_z"],
                )
                for i in range(temp_array.shape[0])
            ]
            data_sf = [rotate_dataset(i, R) for i in data_sf]
            data_sf = np.stack(([i.values for i in data_sf]))

            if isinstance(self.zscale, dict):
                curr_mean = np.asarray(self.zscale.get("mean"))
                curr_std = np.asarray(self.zscale.get("std"))
            else:
                curr_mean = data_sf.mean((0, 1))
                curr_std = data_sf.std((0, 1))
            zscaled_data = (data_sf - curr_mean) / curr_std
            self.zscale_stats = {"mean": curr_mean, "std": curr_std}
            all_arrays = np.concatenate([zscaled_data, all_arrays[:, :, -1:]], axis=-1)

        return all_arrays

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # cropped_arrays = windows[::stride, 0, :, :]
        input_features = self.data[idx, :, :-1]
        output_labels = self.data[idx, :, -1]
        return (
            torch.tensor(input_features, dtype=torch.float32),
            torch.tensor(output_labels, dtype=torch.long),
        )


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
