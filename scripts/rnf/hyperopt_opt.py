import argparse
from functools import partial
from typing import Callable, Dict

import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from text_classification.datamodule import DataModule
from text_classification.datasets import SSTDataset, TextDataset
from text_classification.encoders import RNFEncoder
from text_classification.models import RNF
from text_classification.tokenizers import SpacyTokenizer
from text_classification.vectors import GloVe
from text_classification.vocab import Vocab

def objective(
    params: Dict,
    args: argparse.Namespace,
    datamodule: DataModule,
    embed_mat: np.ndarray,
) -> float:

    seed_everything(42)

    # 6. Setup model
    num_class = 5 if args.fine_grained else 2
    model = RNF(
        input_size=len(datamodule.encoder.vocab),
        num_class=num_class,
        embed_mat=embed_mat,
        filter_width=params["filter_width"],
        embed_dropout=params["embed_dropout"],
        dropout=params["dropout"],
        lr=params["lr"],
    )

    # 7. Setup trainer
    early_stop_callback = EarlyStopping(
        monitor="val_epoch_loss",
        min_delta=0.0001,
        patience=3,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        filepath="./checkpoints/" + "{epoch}",
        save_top_k=1,
        verbose=True,
        monitor="val_epoch_loss",
        mode="min",
    )

    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback],
        gpus=1,
        progress_bar_refresh_rate=0,
        max_epochs=15,
        deterministic=True,
    )

    # 8. Fit model
    trainer.fit(model, ds.train_dataloader(), ds.val_dataloader())

    # 9. Test model
    results = trainer.test(
        test_dataloaders=ds.val_dataloader(), ckpt_path=checkpoint_callback.best_model_path
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
        filter_func = lambda x: x.label != "neutral"
        target_encoding = {"negative": 0, "positive": 1}
    else:
        filter_func = None
        target_encoding = {
            "very negative": 0,
            "negative": 1,
            "neutral": 2,
            "positive": 3,
            "very positive": 4,
        }

    # get data
    train_data, val_data, _ = SSTDataset(
        filter_func=filter_func,
        tokenizer=SpacyTokenizer(),
        train_subtrees=args.train_subtrees,
        fine_grained=args.fine_grained,
    )

    # get vocab
    vocab = Vocab(train_data, min_freq=args.min_freq)

    # get vectors
    vectors = GloVe(name="840B", dim=300)
    embed_mat = vectors.get_matrix(vocab)

    # 4. Setup encoder to encode examples
    encoder = RNFEncoder(vocab=vocab, target_encoding=target_encoding)

    ds = DataModule(
        train=train_data,
        val=val_data,
        encoder=encoder,
        batch_size=64,
    )

    objective = partial(
        objective,
        args=args,
        datamodule=ds,
        embed_mat=embed_mat,
    )

    SPACE = {
        "filter_width": hp.quniform('filter_width', 5, 10),
        "embed_dropout": hp.uniform('embed_dropout', 0.2, 0.4),
        "dropout": hp.uniform("dropout", 0.2, 0.4),
        "lr": hp.uniform("lr", 0.0001, 0.001)
    }

    trials = Trials()
    best = fmin(objective,
        space=SPACE,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials
    )

    print(best)


    