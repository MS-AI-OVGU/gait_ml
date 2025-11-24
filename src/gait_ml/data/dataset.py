from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import bisect

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
        # if self.zscale:
        #     zscaled_data = (
        #         all_arrays[:, :, :3] - all_arrays[:, :, :3].mean((0, 1))
        #     ) / all_arrays[:, :, :3].std((0, 1))
        #     all_arrays = np.concatenate([zscaled_data, all_arrays[:, :, 3:]], axis=-1)
        return all_arrays

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # input_seq = self.data[idx]
        # labels = self.data_label[idx]
        # final_array = np.concatenate([input_seq, np.expand_dims(labels, 1)], axis=1)

        # # Sliding window
        # window_size = self.window_size
        # stride = self.step_size
        # num_dims = 7
        # windows = np.lib.stride_tricks.sliding_window_view(
        #     input_seq, window_shape=(window_size, num_dims)
        # )

        # cropped_arrays = windows[::stride, 0, :, :]
        input_features = self.data[idx, :, :-1]
        output_labels = self.data[idx, :, -1]
        return (
            torch.tensor(input_features, dtype=torch.float32),
            torch.tensor(output_labels, dtype=torch.long),
        )



class TrunkDataset(Dataset):

    label_map = {
        'left': 0,
        'right': 1,
        'ant': 2,
        'post': 3
    }
    def __init__(self, xls_files:list, flexion_type: str, acc_sheet_name: str = "Accelerometer", fs: float = 100.0, cutoff: float = 10):
        
        if flexion_type not in self.label_map:
            raise ValueError(f"Unknown flexion_type: {flexion_type}")

        self.xls_files = xls_files
        self.flexion_type = flexion_type #left #right #ant #post
        self.acc_sheet_name = acc_sheet_name
        self.fs = fs
        self.cutoff = cutoff
        self.data, self.file_offsets = self._load_data()
        self.label = self.label_map[flexion_type]

    def _load_data(self):
        list_of_arrays = []
        file_offsets = []
        cursor = 0

        self.raw_mag = []
        self.raw_acc = []
        self.raw_gyr = []
        self.raw_x = []
        self.raw_y = []

        for file in self.xls_files:
            if file.endswith(".xls"):  
                df_mag = pd.read_excel(file, sheet_name="Accelerometer")      
                df_acc = pd.read_excel(file, sheet_name="Linear Accelerometer")
                df_gyr = pd.read_excel(file, sheet_name="Gyroscope")                
            else:
                raise ValueError("Unsupported file format. Please provide an Excel file.")
        
            # Rename columns
            df_mag = df_mag.rename(columns={
                df_mag.columns[0]: 'time',
                df_mag.columns[1]: 'acc_x',
                df_mag.columns[2]: 'acc_y',
                df_mag.columns[3]: 'acc_z'})
            df_acc = df_acc.rename(columns={
                df_acc.columns[0]: 'time',
                df_acc.columns[1]: 'acc_x',
                df_acc.columns[2]: 'acc_y',
                df_acc.columns[3]: 'acc_z'})
            df_gyr = df_gyr.rename(columns={
                df_gyr.columns[0]: 'time',
                df_gyr.columns[1]: 'gyr_x',
                df_gyr.columns[2]: 'gyr_y',
                df_gyr.columns[3]: 'gyr_z'})

            self.raw_mag.append(df_mag.iloc[:,1:].values)
            self.raw_acc.append(df_acc.iloc[:,1:].values)
            self.raw_gyr.append(df_gyr.iloc[:,1:].values)

            # --- Merge sheets depending on accelerometer type
            if self.acc_sheet_name == "Linear Accelerometer":
                df_merged = pd.merge(df_acc, df_gyr, on='time', how='inner')
            elif self.acc_sheet_name == "Accelerometer":
                df_merged = pd.merge_asof(df_mag.sort_values("time"),
                                          df_gyr.sort_values("time"),
                                          on="time", 
                                          direction="nearest", 
                                          tolerance=0.02 # 20 ms tolerance for 100 Hz data
                                          )
            else:
                raise ValueError(f"Unknown acc_sheet_name: {self.acc_sheet_name}")
            
            arr_gyr = df_merged.iloc[:, 1:].values
            self.raw_x.append(arr_gyr)

             # Filtering: acc and gyr
            sos = butter(4, 10, btype="lowpass", fs=100, output="sos")
            acc = np.stack([sosfiltfilt(sos, df_merged[c].values) for c in ['acc_x','acc_y','acc_z']], axis=-1)
            gyr = np.stack([sosfiltfilt(sos, df_merged[c].values) for c in ['gyr_x','gyr_y','gyr_z']], axis=-1)

             # Channel-wise normalization (acc + gyr, shape Nx6)
            signals = np.concatenate([acc, gyr], axis=-1)
            list_of_arrays.append(signals)

            # --- Track file offsets
            N = len(signals)
            file_offsets.append((cursor, cursor + N))
            cursor += N
        
        all_arrays = np.concatenate(list_of_arrays, axis=0)
        return all_arrays, file_offsets
    
    def get_file_data(self, idx):
        """Return preprocessed data for one specific file index."""
        start, end = self.file_offsets[idx]
        return self.data[start:end, :]
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        x = self.data[idx, :]
        sample = torch.from_numpy(x).float()
        label = torch.tensor(self.label, dtype=torch.long)
        return sample, label


class SitToStandDataset(Dataset):
    
    def __init__(self, xls_files:list, label: str,  acc_sheet_name="Linear Accelerometer", fs: float = 100.0, cutoff: float = 10):

        self.xls_files = xls_files
        self.acc_sheet_name = acc_sheet_name
        self.fs = fs
        self.cutoff = cutoff
        self.label = label
        self.data, self.file_offsets  = self._load_data()

    def _load_data(self):
        list_of_arrays = []
        file_offsets = []
        cursor = 0

        self.raw_mag = []
        self.raw_acc = []
        self.raw_gyr = []
        
        self.raw_x = []
        self.raw_y = []

        for file in self.xls_files:
            if file.endswith(".xls"):  
                df_mag = pd.read_excel(file, sheet_name="Accelerometer")      
                df_acc = pd.read_excel(file, sheet_name="Linear Accelerometer")
                df_gyr = pd.read_excel(file, sheet_name="Gyroscope")                
            else:
                raise ValueError("Unsupported file format. Please provide an Excel file.")
        
            # Rename columns
            df_mag = df_mag.rename(columns={
                df_mag.columns[0]: 'time',
                df_mag.columns[1]: 'acc_x',
                df_mag.columns[2]: 'acc_y',
                df_mag.columns[3]: 'acc_z'})
            df_acc = df_acc.rename(columns={
                df_acc.columns[0]: 'time',
                df_acc.columns[1]: 'acc_x',
                df_acc.columns[2]: 'acc_y',
                df_acc.columns[3]: 'acc_z'})
            df_gyr = df_gyr.rename(columns={
                df_gyr.columns[0]: 'time',
                df_gyr.columns[1]: 'gyr_x',
                df_gyr.columns[2]: 'gyr_y',
                df_gyr.columns[3]: 'gyr_z'})
            #convert radians to degrees
            df_gyr[['gyr_x','gyr_y','gyr_z']] = df_gyr[['gyr_x','gyr_y','gyr_z']] * 180.0 / np.pi

            self.raw_mag.append(df_mag.iloc[:,1:].values)
            self.raw_acc.append(df_acc.iloc[:,1:].values)
            self.raw_gyr.append(df_gyr.iloc[:,1:].values)

            if self.acc_sheet_name == "Linear Accelerometer":
                # NOTE: Merge on time for linear acceleration and gyroscop
                df_merged = pd.merge(df_acc, df_gyr, on='time', how='inner')

            elif self.acc_sheet_name == "Accelerometer":
                # NOTE: Merge on time for acceleration (with g) and gyroscope.
                df_merged = pd.merge_asof(df_mag.sort_values("time"),
                                    df_gyr.sort_values("time"),
                                    on="time",
                                    direction="nearest",
                                    tolerance=0.02  # 20 ms tolerance for 100 Hz data
                                    )
            else:
                raise ValueError(f"Unknown acc_sheet_name: {self.acc_sheet_name}")
            
            arr_raw = df_merged.iloc[:, 1:].values
            self.raw_x.append(arr_raw)

             # Filtering: acc and gyr
            sos = butter(3, 5, btype="lowpass", fs=100, output="sos")
            acc = np.stack([sosfiltfilt(sos, df_merged[c].values) for c in ['acc_x','acc_y','acc_z']], axis=-1)
            gyr = np.stack([sosfiltfilt(sos, df_merged[c].values) for c in ['gyr_x','gyr_y','gyr_z']], axis=-1)

            # Channel-wise normalization (acc + gyr, shape Nx6)
            signals = np.concatenate([acc, gyr], axis=-1)

            list_of_arrays.append(signals)
            
            # Track offsets
            N = len(signals)
            file_offsets.append((cursor, cursor+N))  # <-- store start/end
            cursor += N
                   
        all_arrays = np.concatenate(list_of_arrays, axis=0)
        return all_arrays, file_offsets
    
    def get_file_data(self, idx):
        """Return preprocessed data for one file."""
        start, end = self.file_offsets[idx]
        return self.data[start:end, :]
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        x = self.data[idx, :]
        sample = torch.from_numpy(x).float()
        label = self.label
        return sample, label
    
    
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
