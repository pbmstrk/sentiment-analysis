import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from text_classification import TextClassifier
from text_classification.datamodule import DataModule
from text_classification.datasets import SSTDatasetAlt
from text_classification.encoders import LSTMEncoder
from text_classification.models import NSE
from text_classification.tokenizers import TokenizerSST
from text_classification.vectors import GloVe
from text_classification.vocab import Vocab
from text_classification.utils import get_optimizer, get_scheduler

log = logging.getLogger(__name__)


class LoggingCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        log.info("Epoch: %s", epoch)
        log.info(
            "Training Acc: %.4f\t Training Loss: %.4f",
            metrics["train_epoch_acc"],
            metrics["train_epoch_loss"],
        )
        log.info(
            "Validation Acc: %.4f\t Validation Loss: %.4f",
            metrics["val_epoch_acc"],
            metrics["val_epoch_loss"],
        )


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    log.info("Arguments:\n %s", OmegaConf.to_yaml(cfg))

    seed_everything(42)

    if not cfg.dataset.fine_grained:
        target_encoding = {"negative": 0, "positive": 1}
    else:
        target_encoding = {
            "very negative": 0,
            "negative": 1,
            "neutral": 2,
            "positive": 3,
            "very positive": 4,
        }

    # hydra generates a new working directory for each run
    # want to store data in same directory each run
    root = hydra.utils.to_absolute_path(".data")

    log.info("Downloading data...")
    # 1. Get SST dataset
    train, val, test = SSTDatasetAlt(root=root, tokenizer=TokenizerSST(), **cfg.dataset)

    log.info("Creating vocab...")
    # 2. Get vocab
    vocab = Vocab([train, val, test], **cfg.vocab)

    # 3. Optionally retrieve pre-trained embeddings
    embed_mat = None
    if cfg.vectors.name:
        log.info("Downloading pre-trained word vectors...")
        vectors = GloVe(root=root, name=cfg.vectors.name, dim=300)
        embed_mat = vectors.get_matrix(vocab)

    # 4. Setup encoder to encode examples
    encoder = LSTMEncoder(vocab=vocab, target_encoding=target_encoding)

    # 5. Setup train, val and test dataloaders
    dm = DataModule(
        train=train,
        val=val,
        test=test,
        encoder=encoder,
        batch_size=cfg.datamodule.batch_size,
    )

    # 6. Setup model
    num_class = 5 if cfg.dataset.fine_grained else 2
    model = NSE(
        input_size=len(vocab), num_class=num_class, embed_mat=embed_mat, **cfg.model
    )
    optimizer = get_optimizer(model, **cfg.optimizer)
    scheduler = None
    if hasattr(cfg, "scheduler"):
        scheduler = get_scheduler(optimizer, **cfg.scheduler)

    classifier = TextClassifier(model, optimizer=optimizer, scheduler=scheduler)

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
        callbacks=[LoggingCallback(), early_stop_callback],
        **cfg.trainer
    )
    log.info("Training...")
    # 8. Fit model
    trainer.fit(classifier, dm.train_dataloader(), dm.val_dataloader())

    # 9. Test model
    results = trainer.test(
        test_dataloaders=dm.test_dataloader(),
        ckpt_path=checkpoint_callback.best_model_path,
    )

    log.info(results)


if __name__ == "__main__":
    main()
