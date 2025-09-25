from glob import glob

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    ConfusionMatrix,
    MulticlassPrecision,
    MulticlassRecall,
)

from gait_ml.data.dataset import GaitDataset
from gait_ml.model.layers import Encoder, Seq2Seq


class LitSeq2Seq(L.LightningModule):
    """
    PyTorch Lightning module for the Seq2Seq model.
    Handles training, validation steps, and optimizer configuration.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        dropout_prob,
        learning_rate=1e-3,
        teacher_forcing_ratio=0.5,
        batch_size=8,
        bidirectional=False,
    ):
        super().__init__()
        self.save_hyperparameters()  # Saves all __init__ arguments as hyperparameters

        # Initialize Encoder and Decoder
        encoder = Encoder(
            input_dim, hidden_dim, num_layers, dropout_prob, bidirectional, type="gru"
        )
        # decoder = Decoder(
        #     output_dim, hidden_dim, num_layers, dropout_prob, bidirectional
        # )

        # Initialize the Seq2Seq model
        self.model = Seq2Seq(
            encoder,
            self.device,
            bidirectional,
        )

        # Loss function (CrossEntropyLoss for multiclass classification)
        # class_weight = torch.tensor([0.2, 0.4, 0.4], dtype=torch.float)
        # self.criterion = nn.CrossEntropyLoss(weight=class_weight, reduction="mean")
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.batch_size = batch_size

        # Validation Metrics
        # Initialize metrics for validation
        if output_dim == 1:  # Binary classification
            self.val_precision = BinaryPrecision(
                threshold=0.5
            )  # Adjust threshold as needed
            self.val_recall = BinaryRecall(threshold=0.5)
            self.cm = ConfusionMatrix("binary", num_classes=2)
        else:  # Multiclass classification
            self.val_precision = MulticlassPrecision(
                num_classes=output_dim, average="macro"
            )
            self.val_recall = MulticlassRecall(num_classes=output_dim, average="macro")
            self.cm = ConfusionMatrix("multiclass", num_classes=3)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass of the Seq2Seq model.
        """
        return self.model(src, trg, teacher_forcing_ratio)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        """
        src, trg = batch  # src: input sequence, trg: target sequence

        # Get predictions from the model
        # Use a teacher_forcing_ratio for training
        predictions = self(
            src, trg, teacher_forcing_ratio=self.hparams.teacher_forcing_ratio
        )

        # Calculate loss
        # For CrossEntropyLoss, predictions should be [batch, num_classes, ...], trg should be class indices
        loss = self.criterion(
            predictions.view(-1, self.hparams.output_dim),
            trg.view(-1, self.hparams.output_dim),
        )

        # Log training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        """
        src, trg = batch

        predictions = self(
            src, trg, teacher_forcing_ratio=0.0
        )  # No teacher forcing for validation

        predictions_reshaped = predictions.reshape(-1, self.hparams.output_dim)
        target_reshaped = trg.view(-1, self.hparams.output_dim)

        # Calculate loss
        loss = self.criterion(
            predictions_reshaped.to(target_reshaped.device),
            target_reshaped,
        )

        # For multiclass, get the predicted class by argmax
        predicted_labels = torch.argmax(predictions_reshaped, dim=-1)
        target_labels = torch.argmax(target_reshaped, dim=-1)

        self.val_precision(predicted_labels, target_labels)
        self.val_recall(predicted_labels, target_labels)
        print("\n", self.cm(predicted_labels, target_labels), "\n")

        # Log validation loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,  # LR will reduce after 3 epochs of no val_loss improvement
            ),
            "interval": "epoch",
            "monitor": "val_loss",  # Monitor validation loss for LR reduction
            "strict": False,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def setup(
        self,
        stage: str,
        window_size: int = 256,
        step_size: int = 128,
    ):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_files = glob(
                "/home/geromevivar/projects/gait_ml/data/raw/train/*.xls"
            )

            self.train_dataset = GaitDataset(
                csv_files=train_files,
                window_size=window_size,
                step_size=step_size,
                expand_labels=2,
                acc_sheet_name="Linear Accelerometer",
            )
            val_files = glob("/home/geromevivar/projects/gait_ml/data/raw/val/*.xls")
            self.val_dataset = GaitDataset(
                csv_files=val_files,
                window_size=window_size,
                step_size=window_size,
                expand_labels=2,
                acc_sheet_name="Linear Accelerometer",
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            test_files = glob("/home/geromevivar/projects/gait_ml/data/raw/test/t*.xls")
            self.test_dataset = GaitDataset(
                csv_files=test_files,
                window_size=window_size,
                step_size=window_size,
                expand_labels=2,
                acc_sheet_name="Linear Accelerometer",
            )

        if stage == "predict":
            val_files = glob("/home/geromevivar/projects/gait_ml/data/raw/val/*.xls")
            self.predict_dataset = GaitDataset(
                csv_files=val_files,
                window_size=window_size,
                step_size=window_size,
                expand_labels=2,
                acc_sheet_name="Linear Accelerometer",
            )

    def train_dataloader(self):
        """
        Returns the DataLoader for the training set.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation set.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """
        Returns the DataLoader for the test set.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
