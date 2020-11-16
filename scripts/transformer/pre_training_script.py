import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from text_classification.datamodule import DataModule
from text_classification.datasets import SSTDatasetAlt
from text_classification.tokenizers import TokenizerSST
from text_classification.utils import get_optimizer, get_scheduler

from .pre_training_model import TransformerWithMLMHead
from .pre_training_tokenizer import TransformerEncoderMLM

log = logging.getLogger(__name__)


class LoggingCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        log.info("Epoch: %s", epoch)
        log.info(
            "Training Loss: %.4f",
            metrics["train_epoch_loss"],
        )
        log.info(
            "Validation Loss: %.4f",
            metrics["val_epoch_loss"],
        )

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    log.info("Arguments:\n %s", OmegaConf.to_yaml(cfg))

    seed_everything(42)

    # hydra generates a new working directory for each run
    # want to store data in same directory each run
    root = hydra.utils.to_absolute_path(".data")

    log.info("Downloading data...")
    # 1. Get SST dataset
    train, val, test = SSTDatasetAlt(root=root, tokenizer=TokenizerSST(), **cfg.dataset)

    # 2. Setup encoder
    encoder = TransformerEncoderMLM()
    encoder.add_vocab(
        [train, val, test],
        special_tokens={"cls_token": "<cls>", "sep_token": "<sep>", "mask_token": "<mask>"},
        **cfg.vocab
    )

    # 5. Setup train, val and test dataloaders
    dm = DataModule(
        train=train,
        val=val,
        test=test,
        collate_fn=encoder.collate_fn,
        batch_size=cfg.datamodule.batch_size,
    )

    # 6. Setup model
    model = TransformerWithMLMHead(
        input_size=len(encoder.vocab)
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
        callbacks=[LoggingCallback(), early_stop_callback],
        **cfg.trainer
    )
    log.info("Training...")
    # 8. Fit model
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # 9. Test model
    results = trainer.test(
        test_dataloaders=dm.test_dataloader(),
        ckpt_path=checkpoint_callback.best_model_path,
    )

    log.info(results)


if __name__ == "__main__":
    main()
