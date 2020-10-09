import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from text_classification.models import RNF


def create_fake_data(low, high, dims):

    return torch.randint(low, high, dims)


def create_test_dataloader_rnf(num_examples, batch_size, input_size, seq_len, num_class):
    inputs = create_fake_data(0, input_size, (num_examples, seq_len))
    targets = create_fake_data(0, num_class, (num_examples,))

    dataset = TensorDataset(inputs, targets.long())
    return DataLoader(dataset, batch_size=batch_size)


class TestRNF:

    def test_output_shape(self):
        input_size = 100
        batch_size = 32
        seq_len = 15
        num_class = torch.randint(2, 10, size=(1,)).item()
        inputs = create_fake_data(0, input_size, (batch_size, seq_len))
        targets = None

        batch = (inputs, targets)

        model_args = {
            'input_size': input_size,
            'num_class': num_class
        }

        model = RNF(**model_args)

        assert model(batch).shape == torch.Size([batch_size, num_class])

    @staticmethod
    def _parameters_are_eq(parameter_list_1, parameter_list_2):
        results = []
        for p1, p2 in zip(parameter_list_1, parameter_list_2):
            results.append(torch.all(torch.eq(p1, p2)))
        return all(results)

    def test_forward_backward(self):

        input_size = 100
        num_class = torch.randint(2, 10, size=(1,)).item()

        data_loader = create_test_dataloader_rnf(
            num_examples=100, batch_size=32, input_size=100, seq_len=15, num_class=num_class
        )
        trainer = pl.Trainer(
            progress_bar_refresh_rate=0,
            max_steps=5,
            num_sanity_val_steps=0,
            checkpoint_callback=False,
            logger=False,
        )

        model_args = {
            'input_size': input_size,
            'num_class': num_class
        }

        model_before = RNF(**model_args)
        model_after = RNF(**model_args)
        model_after.load_state_dict(model_before.state_dict())

        assert self._parameters_are_eq(
            list(model_before.parameters()), list(model_after.parameters())
        )

        trainer.fit(model_after, data_loader)

        assert not self._parameters_are_eq(
            list(model_before.parameters()), list(model_after.parameters())
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
    def test_forward_backward_gpu(self):

        input_size = 100
        num_class = torch.randint(2, 10, size=(1,)).item()

        data_loader = create_test_dataloader_rnf(
            num_examples=100, batch_size=32, input_size=100, seq_len=15, num_class=num_class
        )
        trainer = pl.Trainer(
            gpus=1,
            progress_bar_refresh_rate=0,
            max_steps=5,
            num_sanity_val_steps=0,
            checkpoint_callback=False,
            logger=False,
        )

        model_args = {
            'input_size': input_size,
            'num_class': num_class
        }

        model_before = RNF(**model_args)
        model_after = RNF(**model_args)
        model_after.load_state_dict(model_before.state_dict())

        assert self._parameters_are_eq(
            list(model_before.parameters()), list(model_after.parameters())
        )

        trainer.fit(model_after, data_loader)

        assert not self._parameters_are_eq(
            list(model_before.parameters()), list(model_after.parameters())
        )

    def test_weight_freeze(self):

        input_size = 100
        num_class = torch.randint(2, 10, size=(1,)).item()

        data_loader = create_test_dataloader_rnf(
            num_examples=100, batch_size=32, input_size=100, seq_len=15, num_class=num_class
        )
        trainer = pl.Trainer(
            progress_bar_refresh_rate=0,
            max_steps=5,
            num_sanity_val_steps=0,
            checkpoint_callback=False,
            logger=False,
        )

        model_args = {
            'input_size': input_size,
            'num_class': num_class,
            'embed_freeze': True
        }

        model_before = RNF(**model_args)
        model_after = RNF(**model_args)
        model_after.load_state_dict(model_before.state_dict())

        assert torch.all(torch.eq(model_before.embedding.weight, model_after.embedding.weight))

        trainer.fit(model_after, data_loader)

        assert torch.all(torch.eq(model_before.embedding.weight, model_after.embedding.weight))