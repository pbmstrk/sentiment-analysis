import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from text_classification import TextClassifier


class FakeDataset(Dataset):
    def __init__(self, num_input, num_output, return_seq_lengths=False):
        self.num_input = num_input
        self.num_output = num_output
        self.inputs = torch.randint(0, self.num_input, (100, 15))
        self.targets = torch.randint(0, self.num_output, (100,))
        if return_seq_lengths:
            self.seq_lengths = torch.randint(0, 15, (100,))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if hasattr(self, "seq_lengths"):
            return self.inputs[idx], self.targets[idx], self.seq_lengths[idx]
        return self.inputs[idx], self.targets[idx]


class ModelTest:
    def run_model_test(self, trainer_options, model, dataloader):

        trainer = pl.Trainer(**trainer_options)

        classifier = TextClassifier(
            model, scheduler_name="StepLR", scheduler_args={"step_size": 1}
        )

        # check if model actually trains
        # pytorch-lightning/blob/master/tests/base/develop_pipelines.py#L62-L69
        initial_values = torch.tensor(
            [torch.sum(torch.abs(x)) for x in model.parameters()]
        )
        result = trainer.fit(
            classifier, train_dataloader=dataloader, val_dataloaders=dataloader
        )
        post_train_values = torch.tensor(
            [torch.sum(torch.abs(x)) for x in model.parameters()]
        )

        assert result == 1

        assert torch.norm(initial_values - post_train_values) > 0.1

        test_result = trainer.test(classifier, dataloader)

        # test should return something
        assert test_result

    def check_output_shape(self, model, dataloader, expected):

        batch = next(iter(dataloader))
        assert model(batch).shape == expected


class FakeModel(nn.Module):
    def __init__(self, input_size, output_size):

        super().__init__()

        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return x
