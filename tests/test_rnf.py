import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from text_classification.models import RNF

from .base import FakeCNNDataset, ModelTest


class TestRNF(ModelTest):
    def test_output_shape(self):

        # define data
        data = FakeCNNDataset(10, 5)
        dataloader = DataLoader(data, batch_size=32)

        # define model
        model_options = {"input_size": data.num_input, "num_class": data.num_output}
        model = RNF(**model_options)

        # run test
        self.check_output_shape(model, dataloader, torch.Size([32, 5]))

    def test_forward_backward(self):

        # define data
        data = FakeCNNDataset(10, 5)
        dataloader = DataLoader(data, batch_size=32)

        # define trainer options
        trainer_options = {
            "progress_bar_refresh_rate": 0,
            "max_steps": 5,
            "num_sanity_val_steps": 1,
            "checkpoint_callback": False,
            "logger": False,
        }

        # define model
        model_options = {"input_size": data.num_input, "num_class": data.num_output}
        model = RNF(**model_options)

        # run test
        self.run_model_test(trainer_options, model, dataloader)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
    def test_forward_backward_gpu(self):

        # define data
        data = FakeCNNDataset(10, 5)
        dataloader = DataLoader(data, batch_size=32)

        # define trainer options
        trainer_options = {
            "gpus": 1,
            "progress_bar_refresh_rate": 0,
            "max_steps": 5,
            "num_sanity_val_steps": 1,
            "checkpoint_callback": False,
            "logger": False,
        }

        # define model
        model_options = {"input_size": data.num_input, "num_class": data.num_output}
        model = RNF(**model_options)

        # run test
        self.run_model_test(trainer_options, model, dataloader)

    def test_weight_freeze(self):

        data = FakeCNNDataset(10, 5)
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
            "input_size": data.num_input,
            "num_class": data.num_output,
            "freeze_embed": True,
            "embed_mat": embed_mat,
        }

        model_before = RNF(**model_options)
        model_after = RNF(**model_options)
        model_after.load_state_dict(model_before.state_dict())

        # check if embedding weight is correctly loaded and remains fixed
        assert torch.all(
            torch.eq(model_before.embedding.weight, model_after.embedding.weight)
        )
        trainer.fit(model_after, dataloader)
        assert torch.all(
            torch.eq(model_before.embedding.weight, model_after.embedding.weight)
        )
