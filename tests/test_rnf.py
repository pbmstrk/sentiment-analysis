import pytest
import torch
from torch.utils.data import DataLoader

from text_classification.models import RNF

from .base import FakeDataset, ModelTest


class TestRNF(ModelTest):
    def test_output_shape(self):

        # define data
        data = FakeDataset(10, 5)
        dataloader = DataLoader(data, batch_size=32)

        # define model
        model_options = {"input_size": data.num_input, "num_class": data.num_output}
        model = RNF(**model_options)

        # run test
        self.check_output_shape(model, dataloader, torch.Size([32, 5]))

    def test_one_element(self):

        # define data
        data = FakeDataset(10, 5)
        dataloader = DataLoader(data, batch_size=1)

        # define model
        model_options = {"input_size": data.num_input, "num_class": data.num_output}
        model = RNF(**model_options)

        # run test
        self.check_output_shape(model, dataloader, torch.Size([1, 5]))

    def test_forward_backward(self):

        # define data
        data = FakeDataset(10, 5)
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
        data = FakeDataset(10, 5)
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


