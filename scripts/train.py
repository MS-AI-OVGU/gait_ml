from datetime import datetime
from glob import glob

import lightning as pl
import numpy as np
import pandas as pd
import wandb
from gait_ml.data.datamodule import GaitDataModule
from gait_ml.model.litmodel import LitSeq2Seq
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold, train_test_split


def main():
    all_files = glob(
        "/home/geromevivar/projects/spine_interaction/data/dataset2/10092025/imu/Mobilephone/Termin_1_Vicon/ID_*_T1/*_1_2mW_IPhone.xls",
        recursive=True,
    )
    all_files = np.sort(all_files)
    print(f"Processing: {len(all_files)} samples")

    ids = [int(i.split("/")[-1].split("_")[0]) for i in all_files]
    window_size = 256
    step_size = 128
    batch_size = 256
    expand_labels = 1

    group_df = pd.read_csv(
        "/home/geromevivar/projects/gait_ml/data/processed/groups.csv", index_col="ID"
    )
    group_df.columns = ["group", "status"]
    group_df = group_df[group_df.group.notna()]
    # group_df.drop(index=[6, 22], inplace=True)

    # Healthy group = 0 | Pain group = 1
    group_df.replace("h", 0, inplace=True)
    group_df.replace("p", 1, inplace=True)
    grouping = group_df.loc[ids].group.values
    grouping

    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store scores
    validation_scores = []
    test_scores = []

    X = np.arange(len(ids)).reshape(-1, 1)
    y = grouping

    # Outer loop for K-Fold cross-validation
    # This loop creates the primary TEST set for each fold.
    for fold, (train_val_index, test_index) in enumerate(skf.split(X, y)):
        print(f"=============== FOLD {fold + 1}/{n_splits} ================")

        # Split data into a temporary training+validation set and the final test set
        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            stratify=y_train_val,
            random_state=1,
        )
        print("X_train: ", X_train.squeeze())
        print("X_val: ", X_val.squeeze())
        print("test_index: ", test_index)

        data_module = GaitDataModule(
            all_files,
            batch_size=batch_size,
            window_size=window_size,
            step_size=step_size,
            train_idx=X_train.squeeze(),
            val_idx=X_val.squeeze(),
            test_idx=test_index,
            expand_labels=expand_labels,
            acc_sheet_name="Linear Accelerometer",
            num_workers=16,
        )

        # Define model parameters
        INPUT_DIM = 6  # As specified by the user
        OUTPUT_DIM = 3  # Changed to 1 as requested
        HIDDEN_DIM = 64
        NUM_LAYERS = 2
        DROPOUT_PROB = 0.5
        LEARNING_RATE = 0.001
        TEACHER_FORCING_RATIO = 0.5
        NUM_EPOCHS = 250

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"GRU-expandlabel{expand_labels}_{current_time}"
        project_name = "backpain"
        print(f"Starting run: {run_name} in {project_name}")

        # Initialize the Lightning Module
        model = LitSeq2Seq(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout_prob=DROPOUT_PROB,
            learning_rate=LEARNING_RATE,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO,
        )

        wandb_logger = WandbLogger(
            project=project_name,
            name=run_name,
            log_model="all",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_f1score",  # Metric to monitor
            mode="max",  # 'min' for loss, 'max' for accuracy
            save_top_k=3,  # Save the top 3 models
            dirpath=f"{project_name}/{run_name}/checkpoints/",  # Directory to save checkpoints
            filename="model-{epoch:02d}-{val_f1score:.2f}",  # Checkpoint file name
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator="gpu",
            devices=1,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor],
            # logger=pl.pytorch.loggers.TensorBoardLogger("tb_logs", name="retrain_seq2seq_model") # Uncomment for TensorBoard logging
        )

        print("Starting training...")
        # Train the model
        data_module.setup("train")
        trainer.fit(
            model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )

        print("\nTraining complete!")
        wandb.finish()
        break


if __name__ == "__main__":
    main()
