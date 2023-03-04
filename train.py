"""XXX
"""
import argparse
from pathlib import Path
import pickle
import gzip

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch.utils.data import DataLoader, Dataset
import wandb

import artifact
import model


class MNISTDataset(Dataset):
    def __init__(self, path: Path, *, transform=None, target_transform=None):
        """XXX
        @param path: path to artifact pkl.gz
        """
        with gzip.open(path, "rb") as f:
            self.x, self.y = pickle.load(f)
        assert len(self.x) == len(self.y)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple:
        x, y = self.x[idx, :], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class LitMnistModel(pl.LightningModule):
    def __init__(
        self,
        *,
        learning_rate: float = 3e-4,
        fc1_layers: int = 1024,
        fc2_layers: int = 128,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.model = model.MnistModel(
            fc1_layers=fc1_layers, fc2_layers=fc2_layers, dropout_p=dropout_p
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        x, y, logits, loss = self._run_batch(batch)
        self.log("train/loss", loss)
        outputs = {"loss": loss, "logits": logits}
        return outputs

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        x, y, logits, loss = self._run_batch(batch)
        self.log("validation/loss", loss)
        outputs = {"loss": loss, "logits": logits}
        return outputs

    def _run_batch(self, batch: tuple[torch.tensor, torch.tensor]) -> tuple:
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        return x, y, logits, loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--learning_rate", type=float, default=3e-4, help="learning rate"
        )
        parser.add_argument("--fc1_layers", type=int, default=1024)
        parser.add_argument("--fc2_layers", type=int, default=128)
        parser.add_argument("--dropout_p", type=float, default=0.2)
        return parser


class PredictionCallback(pl.Callback):
    """Write predictions from validation set..."""

    def __init__(self, max_images: int = 32):
        super().__init__()
        self.max_images = 32

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        module: pl.LightningModule,
        outputs: dict,
        batch: tuple[torch.tensor, torch.tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pass
        # if not outputs:
        #     return

        # x, y = batch
        # ypred = outputs['logits'].argmax(dim=1)
        # trainer.logger.log_table('validation/predictions',
        #                          columns=['label', 'prediction'],
        #                          data=list(zip(*[y.tolist(),
        #                                          ypred.tolist()])))

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, module: pl.LightningModule
    ) -> None:
        pass

    @rank_zero_only
    def on_train_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        print("\n\nTRAIN END\n\n")


def main():
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--epochs", type=int, default=2, help="maximum number of epochs"
    )
    parser.add_argument("--seed", type=int, default=42, help="global seed")
    parser.add_argument(
        "--dataset_alias", type=str, default="latest", help="alias to dataset artifact"
    )

    LitMnistModel.add_to_argparse(parser)

    args = parser.parse_args()

    run = wandb.init(project="mnist", entity=None, job_type="train", config=args)

    pl.seed_everything(wandb.config["seed"])

    alias = wandb.config["dataset_alias"]
    print(f"downloading artifact {alias}...")
    data_artifact = run.use_artifact(f"{artifact.ARTIFACT}:{alias}")
    path = Path(data_artifact.download())

    train_dataset = MNISTDataset(
        path / artifact.TRAIN_NAME,
        transform=torch.tensor,
        target_transform=torch.tensor,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=2,
        batch_size=wandb.config["batch_size"],
    )
    val_dataset = MNISTDataset(
        path / artifact.VALIDATION_NAME,
        transform=torch.tensor,
        target_transform=torch.tensor,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=2, batch_size=64)

    lit_model = LitMnistModel(
        learning_rate=wandb.config["learning_rate"],
        fc1_layers=wandb.config["fc1_layers"],
        fc2_layers=wandb.config["fc2_layers"],
        dropout_p=wandb.config["dropout_p"],
    )

    logger = pl.loggers.WandbLogger(log_model="all")
    logger.watch(lit_model, log_freq=100)

    # Save a model checkpoint on validation/loss.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        monitor="validation/loss",
        mode="min",
        dirpath=logger.experiment.dir,
        filename="epoch-{epoch:03d}.validation_loss-{validation/loss:.5f}",
        auto_insert_metric_name=False,
    )

    # Generate a summary of all model layers on fit start.
    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    # Save predictions.
    prediction_callback = PredictionCallback()

    trainer = pl.Trainer(
        max_epochs=wandb.config["epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, summary_callback, prediction_callback],
    )

    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    run.finish()


if __name__ == "__main__":
    main()
