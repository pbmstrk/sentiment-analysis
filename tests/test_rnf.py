import torch
import pytest
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np

from text_classification.models import RNF


class FakeRNFDataset(Dataset):

    def __init__(self):
        self.inputs = torch.randint(0, 10, (100, 15))
        self.targets = torch.randint(0, 10, (100, ))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class TestRNF:

    def test_output_shape(self):

        data = FakeRNFDataset()
        dataloader = DataLoader(data, batch_size=32)

        batch = next(iter(dataloader))

        model_options = {
            'input_size': 100,
            'num_class': 10
        }

        model = RNF(**model_options)

        assert model(batch).shape == torch.Size([32, 10])


    def run_model_test(self, trainer_options, model):

        data = FakeRNFDataset()
        dataloader = DataLoader(data, batch_size=32)

        trainer = pl.Trainer(**trainer_options)

        # check if model actually trains
        # pytorch-lightning/blob/master/tests/base/develop_pipelines.py#L62-L69
        initial_values = torch.tensor([torch.sum(torch.abs(x)) for x in model.parameters()])
        result = trainer.fit(model, train_dataloader=dataloader, val_dataloaders=dataloader)
        post_train_values = torch.tensor([torch.sum(torch.abs(x)) for x in model.parameters()])

        assert result == 1

        assert torch.norm(initial_values - post_train_values) > 0.1

        test_result = trainer.test(model, dataloader)

    def test_forward_backward(self):
        
        trainer_options = {
            "progress_bar_refresh_rate": 0,
            "max_steps": 5,
            "num_sanity_val_steps": 1,
            "checkpoint_callback": False,
            "logger": False,
        }

        model_options = {
            'input_size': 100,
            'num_class': 10
        }

        model = RNF(**model_options)

        self.run_model_test(trainer_options, model)

        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
    def test_forward_backward_gpu(self):

        trainer_options = {
            "gpus": 1,
            "progress_bar_refresh_rate": 0,
            "max_steps": 5,
            "num_sanity_val_steps": 1,
            "checkpoint_callback": False,
            "logger": False,
        }

        model_options = {
            'input_size': 100,
            'num_class': 10
        }

        model = RNF(**model_options)

        self.run_model_test(trainer_options, model)

    def test_weight_freeze(self):

        data = FakeRNFDataset()
        dataloader = DataLoader(data, batch_size=32)
        
        trainer_options = {
            "progress_bar_refresh_rate": 0,
            "max_steps": 5,
            "num_sanity_val_steps": 0,
            "checkpoint_callback": False,
            "logger": False,
        }

        trainer = pl.Trainer(**trainer_options)

        embed_mat = np.random.randn(100, 300)

        model_options = {
            'input_size': 100,
            'num_class': 10,
            'freeze_embed': True,
            'embed_mat': embed_mat
        }

        model_before = RNF(**model_options)
        model_after = RNF(**model_options)
        model_after.load_state_dict(model_before.state_dict())

        # check if embedding weight is correctly loaded and remains fixed
        assert torch.all(torch.eq(model_before.embedding.weight, model_after.embedding.weight))
        trainer.fit(model_after, dataloader)
        assert torch.all(torch.eq(model_before.embedding.weight, model_after.embedding.weight))
