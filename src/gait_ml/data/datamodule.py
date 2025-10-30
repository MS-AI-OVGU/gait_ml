import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Subset
from .dataset import GaitDataset  # Your custom dataset
from glob import glob

import torch


class GaitDataModule(L.LightningDataModule):
    def __init__(
        self,
        all_files,
        batch_size=32,
        window_size=128,
        step_size=64,
        train_idx=None,
        val_idx=None,
        test_idx=None,
        expand_labels=0,
        acc_sheet_name="Linear Accelerometer",
        num_workers=4,
    ):
        super().__init__()
        self.all_files = all_files
        self.batch_size = batch_size
        self.window_size = window_size
        self.step_size = step_size
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.expand_labels = expand_labels
        self.acc_sheet_name = acc_sheet_name
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "train" and self.train_idx is not None and self.val_idx is not None:
            self.train_dataset = GaitDataset(
                csv_files=self.all_files[self.train_idx],
                window_size=self.window_size,
                step_size=self.step_size,
                expand_labels=self.expand_labels,
                acc_sheet_name=self.acc_sheet_name,
                gyr_sheet_name="Gyroscope",
            )
            self.val_dataset = GaitDataset(
                csv_files=self.all_files[self.val_idx],
                window_size=self.window_size,
                step_size=self.window_size,
                expand_labels=self.expand_labels,
                acc_sheet_name=self.acc_sheet_name,
                gyr_sheet_name="Gyroscope",
            )

        if stage == "val" and self.val_idx is not None:
            self.val_dataset = GaitDataset(
                csv_files=self.all_files[self.val_idx],
                window_size=self.window_size,
                step_size=self.window_size,
                expand_labels=self.expand_labels,
                acc_sheet_name=self.acc_sheet_name,
                gyr_sheet_name="Gyroscope",
            )

        if stage == "test" and self.test_idx is not None:
            self.test_dataset = GaitDataset(
                csv_files=self.all_files[self.test_idx],
                window_size=self.window_size,
                step_size=self.window_size,
                expand_labels=self.expand_labels,
                acc_sheet_name=self.acc_sheet_name,
                gyr_sheet_name="Gyroscope",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # collate_fn=self.concatenate_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # collate_fn=self.concatenate_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # collate_fn=self.concatenate_collate_fn,
        )

    def concatenate_collate_fn(self, batch):
        """Collates a batch by concatenating sequences along the time dimension (dim=0)."""
        sequences, targets = zip(*batch)
        concatenated_sequences = torch.cat(sequences, dim=0)
        concatenated_targets = torch.cat(targets, 0)
        return concatenated_sequences, concatenated_targets


if __name__ == "__main__":
    # Example usage of GaitDataModule
    all_files = glob(
        "/home/geromevivar/projects/spine_interaction/data/dataset2/10092025/imu/Mobilephone/Termin_1_Vicon/ID_*_T1/*_1_2mW_IPhone.xls",
        recursive=True,
    )  # Adjust path as needed
    batch_size = 32

    # Initialize the GaitDataModule
    data_module = GaitDataModule(all_files=all_files, batch_size=batch_size)

    # Setup the data module for the "fit" stage
    data_module.train_idx = list(
        range(0, int(0.8 * len(all_files)))
    )  # 80% for training
    data_module.val_idx = list(
        range(int(0.8 * len(all_files)), len(all_files))
    )  # 20% for validation
    data_module.setup(stage="fit")

    # Access the dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Iterate through the training data
    for batch in train_loader:
        inputs, targets = batch
        print("Inputs:", inputs)
        print("Targets:", targets)
        break  # Just to show one batch
