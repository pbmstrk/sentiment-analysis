import argparse
from functools import partial
from typing import Callable

import numpy as np
import optuna
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from text_classification import TextClassifier
from text_classification.datamodule import DataModule
from text_classification.datasets import SSTDatasetAlt, TextDataset
from text_classification.encoders import CNNEncoder
from text_classification.models import RNF
from text_classification.tokenizers import TokenizerSST
from text_classification.vectors import GloVe
from text_classification.vocab import Vocab


class OptunaCallback(Callback):
    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:

        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        current_score = metrics.get(self.monitor)
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


def objective(
    trial: optuna.trial.Trial,
    args: argparse.Namespace,
    train: TextDataset,
    val: TextDataset,
    encoder: Callable,
    embed_mat: np.ndarray,
) -> float:

    seed_everything(42)

    # 5. Setup train, val and test dataloaders
    dm = DataModule(
        train=train,
        val=val,
        encoder=encoder,
        batch_size=64,
    )

    # 6. Setup model
    num_class = 5 if args.fine_grained else 2
    model = RNF(
        input_size=len(vocab),
        num_class=num_class,
        embed_mat=embed_mat,
        filter_width=trial.suggest_int("filter_width", 5, 8),
        embed_dropout=trial.suggest_float("embed_dropout", 0.2, 0.4, step=0.05),
        dropout=trial.suggest_float("dropout", 0.2, 0.4, step=0.05),
    )
    classifier = TextClassifier(model, optimizer_args={"lr": trial.suggest_float("lr", 0.0001, 0.001, step=0.00005)})
    

    # 7. Setup trainer
    early_stop_callback = EarlyStopping(
        monitor="val_epoch_loss",
        min_delta=0.0001,
        patience=3,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        filepath="./checkpoints/" + "trial_{}".format(trial.number) + "{epoch}",
        save_top_k=1,
        verbose=True,
        monitor="val_epoch_loss",
        mode="min",
    )

    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        callbacks=[OptunaCallback(trial, "val_epoch_loss"), early_stop_callback],
        gpus=1,
        progress_bar_refresh_rate=0,
        max_epochs=15,
        deterministic=True,
    )

    # 8. Fit model
    trainer.fit(classifier, dm.train_dataloader(), dm.val_dataloader())

    # 9. Test model
    results = trainer.test(
        test_dataloaders=dm.val_dataloader(),
        ckpt_path=checkpoint_callback.best_model_path,
    )

    # not actually results on test set - key stems from test_epoch_end
    return results[0]["test_epoch_loss"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_grained", action="store_true")
    parser.add_argument("--train_subtrees", action="store_true")
    parser.add_argument("--min_freq", default=1, type=int)
    args = parser.parse_args()
    print(args)

    # define filter function and target encoding
    if not args.fine_grained:
        target_encoding = {"negative": 0, "positive": 1}
    else:
        target_encoding = {
            "very negative": 0,
            "negative": 1,
            "neutral": 2,
            "positive": 3,
            "very positive": 4,
        }

    # get data
    train_data, val_data, _ = SSTDatasetAlt(
        tokenizer=TokenizerSST(),
        train_subtrees=args.train_subtrees,
        fine_grained=args.fine_grained,
    )

    # get vocab
    vocab = Vocab([train_data, val_data], min_freq=args.min_freq)

    # get vectors
    vectors = GloVe(name="840B", dim=300)
    embed_mat = vectors.get_matrix(vocab)

    # 4. Setup encoder to encode examples
    encoder = CNNEncoder(vocab=vocab, target_encoding=target_encoding)

    objective = partial(
        objective,
        args=args,
        train=train_data,
        val=val_data,
        encoder=encoder,
        embed_mat=embed_mat,
    )

    pruner = optuna.pruners.PercentilePruner(0.5, n_warmup_steps=7)

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=30, timeout=None)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
