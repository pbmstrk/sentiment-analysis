import hydra
from omegaconf import DictConfig
from text_classification.datamodule import DataModule
from text_classification.datasets import SSTDataset
from text_classification.encoders import RNFEncoder
from text_classification.models import RNF
from text_classification.vectors import GloVe
from text_classification.vocab import Vocab
from text_classification.tokenizers import SpacyTokenizer

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback



class LoggingCallback(Callback):

    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        print(f"Epoch: {epoch}")
        print(f"Training Acc: {metrics['epoch_train_acc']:.4f}\t Training Loss: {metrics['epoch_train_loss']:.4f}")
        print(f"Validation Acc: {metrics['epoch_val_acc']:.4f}\t Validation Loss: {metrics['epoch_val_loss']:.4f}\n")

early_stop_callback = EarlyStopping(
   monitor='epoch_val_loss',
   min_delta=0.0001,
   patience=3,
   verbose=False,
   mode='min'
)

checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/'+'{epoch}',
    save_top_k=1,
    verbose=False,
    monitor='epoch_val_loss',
    mode='min'
)

@hydra.main(config_name="config")
def main(cfg: DictConfig):

    seed_everything(42)

    if not cfg.dataset.fine_grained:
        filter_func = lambda x: x.label != 'neutral'
    else:
        filter_func = None

    # hydra generates a new working directory for each run
    # want to store data in same directory each run
    root=hydra.utils.to_absolute_path('.data')
    
    # 1. Get SST dataset
    train, val, test = SSTDataset(root=root, filter_func=filter_func,
                        tokenizer=SpacyTokenizer(), **cfg.dataset)

    # 2. Get vocab
    vocab = Vocab(train, **cfg.vocab)

    # 3. Retrieve pre-trained embeddings
    vectors = GloVe(root=root, name='840B', dim=300)
    embed_mat = vectors.get_matrix(vocab)
    
    # 4. Setup encoder to encode examples
    encoder = RNFEncoder(vocab=vocab, target_encoding={"negative": 0, "positive": 1})

    # 5. Setup train, val and test dataloaders
    ds = DataModule(train=train, val=val, test=test, encoder=encoder,
                batch_size=cfg.datamodule.batch_size)

    # 6. Setup model
    num_class = 5 if cfg.dataset.fine_grained else 2
    model = RNF(input_size=len(vocab), num_class=num_class, embed_mat=embed_mat, **cfg.model)

    # 7. Setup trainer
    trainer = Trainer(early_stop_callback=early_stop_callback,
                     checkpoint_callback=checkpoint_callback,
                     callbacks=[LoggingCallback()], **cfg.trainer)

    # 8. Fit model
    trainer.fit(model, ds.train_dataloader(), ds.val_dataloader())

    # 9. Test model
    trainer.test(model, ds.test_dataloader(), ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
