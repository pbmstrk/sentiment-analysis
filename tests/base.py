import pytorch_lightning as pl
import torch

class ModelTest:

    def run_model_test(self, trainer_options, model, dataloader):

        trainer = pl.Trainer(**trainer_options)

        # check if model actually trains
        # pytorch-lightning/blob/master/tests/base/develop_pipelines.py#L62-L69
        initial_values = torch.tensor([torch.sum(torch.abs(x)) for x in model.parameters()])
        result = trainer.fit(model, train_dataloader=dataloader, val_dataloaders=dataloader)
        post_train_values = torch.tensor([torch.sum(torch.abs(x)) for x in model.parameters()])

        assert result == 1

        assert torch.norm(initial_values - post_train_values) > 0.1

        test_result = trainer.test(model, dataloader)

        # test should return something
        assert test_result

    def check_output_shape(self, model, dataloader, expected):

        batch = next(iter(dataloader))
        assert model(batch).shape == expected

