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
from hydra import initialize, compose
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="train_config", version_base="1.3")
def main(config: DictConfig):
    # with initialize(config_path="../configs", job_name="train", version_base="1.3"):
    #     config = compose(config_name="train_config")
    pl.seed_everything(config.general.random_state)
    all_files = glob(config.general.data_path, recursive=True)
    all_files = np.sort(all_files)
    print(f"Processing: {len(all_files)} samples")

    ids = [int(i.split("/")[-1].split("_")[0]) for i in all_files]
    group_df = pd.read_csv(config.general.group_file, index_col="ID")
    group_df.columns = ["group", "status"]
    group_df = group_df[group_df.group.notna()]
    group_df.replace("h", 0, inplace=True)
    group_df.replace("p", 1, inplace=True)
    grouping = group_df.loc[ids].group.values

    skf = StratifiedKFold(
        n_splits=config.general.n_splits,
        shuffle=True,
        random_state=config.general.random_state,
    )

    # Outer loop for K-Fold cross-validation
    # This loop creates the primary TEST set for each fold.
    for fold, (train_val_index, test_index) in enumerate(
        skf.split(np.arange(len(ids)).reshape(-1, 1), grouping)
    ):
        print(
            f"=============== FOLD {fold + 1}/{config.general.n_splits} ================"
        )
        print("current_test set:", test_index)

        # Split data into a temporary training+validation set and the final test set
        X_train_val, X_test = (
            np.arange(len(ids))[train_val_index],
            np.arange(len(ids))[test_index],
        )
        y_train_val, y_test = grouping[train_val_index], grouping[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            stratify=y_train_val,
            random_state=1,
        )

        data_module = GaitDataModule(
            all_files,
            batch_size=config.data.batch_size,
            window_size=config.data.window_size,
            step_size=config.data.step_size,
            train_idx=X_train.squeeze(),
            val_idx=X_val.squeeze(),
            test_idx=test_index,
            expand_labels=config.data.expand_labels,
            acc_sheet_name=config.data.acc_sheet_name,
            num_workers=config.data.num_workers,
        )

        model = LitSeq2Seq(
            input_dim=config.model.input_dim,
            output_dim=config.model.output_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout_prob=config.model.dropout_prob,
            learning_rate=config.model.learning_rate,
            teacher_forcing_ratio=config.model.teacher_forcing_ratio,
        )

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = (
            f"Fold{fold + 1}-GRU-expandlabel{config.data.expand_labels}_{current_time}"
        )
        wandb_logger = WandbLogger(
            project=config.general.project_name,
            name=run_name,
            log_model="all",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor=config.training.monitor_metric,
            mode=config.training.monitor_mode,
            save_top_k=config.training.save_top_k,
            dirpath=f"{config.general.project_name}/{run_name}/checkpoints/",
            filename="model-{epoch:02d}-{val_f1score:.2f}",
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(
            max_epochs=config.training.num_epochs,
            accelerator=config.training.accelerator,
            devices=config.training.devices,
            log_every_n_steps=config.training.log_every_n_steps,
            check_val_every_n_epoch=config.training.check_val_every_n_epoch,
            enable_progress_bar=True,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor],
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


if __name__ == "__main__":
    main()
