import pytest
import torch

from text_classification import TextClassifier

from .base import FakeModel


class TestTextClassifier:
    def test_optimizer_input(self):
        # with correct name
        TextClassifier(FakeModel(5, 2), optimizer_name="Adam")

        # with incorrect name
        with pytest.raises(AttributeError):
            TextClassifier(FakeModel(5, 2), optimizer_name="Ada")

    def test_scheduler_input(self):
        # with correct name and args
        TextClassifier(
            FakeModel(5, 2), scheduler_name="StepLR", scheduler_args={"step_size": 1}
        )

        # with incorrect name
        with pytest.raises(AttributeError):
            TextClassifier(FakeModel(5, 2), scheduler_name="StepL")

        # with correct name and missing args
        with pytest.raises(AttributeError):
            TextClassifier(FakeModel(5, 2), scheduler_name="StepLR")
