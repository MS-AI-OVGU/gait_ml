import lightning as L
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    ConfusionMatrix,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

from gait_ml.model.layers import Encoder, Seq2Seq


class LitSeq2Seq(L.LightningModule):
    """
    PyTorch LightningModule for the Seq2Seq model.
    Handles training, validation, and optimizer configuration.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_prob: float,
        learning_rate: float = 1e-3,
        teacher_forcing_ratio: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the Seq2Seq model
        encoder = Encoder(
            input_dim,
            hidden_dim,
            num_layers,
            dropout_prob,
            bidirectional=False,
            type="gru",
        )
        self.model = Seq2Seq(encoder, self.device, bidirectional=False)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # Metrics
        if output_dim == 1:
            self.val_precision = BinaryPrecision(threshold=0.5)
            self.val_recall = BinaryRecall(threshold=0.5)
            self.cm = ConfusionMatrix("binary", num_classes=2)
        else:
            self.val_precision = MulticlassPrecision(
                num_classes=output_dim, average="macro"
            )
            self.val_recall = MulticlassRecall(num_classes=output_dim, average="macro")
            self.val_f1score = MulticlassF1Score(
                num_classes=output_dim, average="macro"
            )
            self.cm = ConfusionMatrix("multiclass", num_classes=output_dim)

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.5):
        return self.model(src, trg, teacher_forcing_ratio)

    def training_step(self, batch, batch_idx):
        src, trg = batch
        predictions = self(
            src, trg, teacher_forcing_ratio=self.hparams.teacher_forcing_ratio
        )
        loss = self.criterion(
            predictions.view(-1, self.hparams.output_dim), trg.view(-1)
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        predictions = self(src, trg, teacher_forcing_ratio=0.0)
        predictions_reshaped = predictions.reshape(-1, self.hparams.output_dim)
        target_reshaped = trg.view(-1)
        loss = self.criterion(
            predictions_reshaped.to(target_reshaped.device),
            target_reshaped,
        )
        # For multiclass, get the predicted class by argmax
        predicted_labels = torch.argmax(predictions_reshaped, dim=-1)

        self.val_precision(predicted_labels, target_reshaped)
        self.val_recall(predicted_labels, target_reshaped)
        self.val_f1score(predicted_labels, target_reshaped)
        print("\n", self.cm(predicted_labels, target_reshaped), "\n")

        loss = self.criterion(
            predictions.view(-1, self.hparams.output_dim), trg.view(-1)
        )
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
        self.log(
            "val_f1score",
            self.val_f1score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def pred_step(self, batch):
        self.model(src, trg, teacher_forcing_ratio)
        predictions = self(src, trg, teacher_forcing_ratio=0.0)
        predictions_reshaped = predictions.reshape(-1, self.hparams.output_dim)
        target_reshaped = trg.view(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 5,
            "strict": False,
        }


# Example usage of LitSeq2Seq for multiclass classification training
if __name__ == "__main__":
    import torch
    from lightning import Trainer

    # Example hyperparameters (adjust as needed)
    input_dim = 3  # Number of input features per timestep
    output_dim = 3  # Number of classes for classification
    hidden_dim = 6
    num_layers = 2
    dropout_prob = 0.3
    learning_rate = 1e-5
    teacher_forcing_ratio = 0.5
    batch_size = 32

    # Instantiate the LightningModule
    model = LitSeq2Seq(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        learning_rate=learning_rate,
        teacher_forcing_ratio=teacher_forcing_ratio,
        batch_size=batch_size,
    )

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(max_epochs=10, accelerator="auto")

    # Train the model
    trainer.fit(model)

    # For testing after training:
    # model.setup(stage="test")
    # trainer.test(model, model.test_dataloader())
