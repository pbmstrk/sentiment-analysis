import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from text_classification.models import AttentionLSTM


def create_fake_data(low, high, dims):

    return torch.randint(low, high, dims)


def create_test_dataloader_lstm(num_examples, batch_size, input_size, seq_len):
    inputs = create_fake_data(0, input_size, (num_examples, seq_len))
    targets = create_fake_data(0, 2, (num_examples,))
    seqlengths = seq_len * torch.ones((num_examples,))

    dataset = TensorDataset(inputs, targets.float(), seqlengths)
    return DataLoader(dataset, batch_size=batch_size)


class TestAttentionLSTM:
    def test_output_shape(self):
        input_size = 100
        batch_size = 32
        seq_len = 15
        inputs = create_fake_data(0, input_size, (batch_size, seq_len))
        targets = None
        seqlengths = create_fake_data(5, seq_len, (batch_size - 1,))
        # need to ensure we have sequence of max_len
        seqlengths = torch.cat((torch.tensor([seq_len]), seqlengths))

        batch = (inputs, targets, seqlengths)

        model = AttentionLSTM(input_size=input_size)

        assert model(batch).shape == torch.Size([batch_size])

    @staticmethod
    def _parameters_are_eq(parameter_list_1, parameter_list_2):
        results = []
        for p1, p2 in zip(parameter_list_1, parameter_list_2):
            results.append(torch.all(torch.eq(p1, p2)))
        return all(results)

    def test_forward_backward(self):

        data_loader = create_test_dataloader_lstm(
            num_examples=100, batch_size=32, input_size=100, seq_len=15
        )
        trainer = pl.Trainer(
            progress_bar_refresh_rate=0,
            max_steps=5,
            num_sanity_val_steps=0,
            checkpoint_callback=False,
            logger=False,
        )

        model_before = AttentionLSTM(input_size=100)
        model_after = AttentionLSTM(input_size=100)
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

        data_loader = create_test_dataloader_lstm(
            num_examples=100, batch_size=32, input_size=100, seq_len=15
        )
        trainer = pl.Trainer(
            gpus=1,
            progress_bar_refresh_rate=0,
            max_steps=5,
            num_sanity_val_steps=0,
            checkpoint_callback=False,
            logger=False,
        )

        model_before = AttentionLSTM(input_size=100)
        model_after = AttentionLSTM(input_size=100)
        model_after.load_state_dict(model_before.state_dict())

        assert self._parameters_are_eq(
            list(model_before.parameters()), list(model_after.parameters())
        )

        trainer.fit(model_after, data_loader)

        assert not self._parameters_are_eq(
            list(model_before.parameters()), list(model_after.parameters())
        )
