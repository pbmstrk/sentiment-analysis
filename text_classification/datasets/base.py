from dataclasses import dataclass
from typing import List, Union

from torch.utils.data import Dataset


@dataclass
class Example:
    text: List[str]
    label: Union[int, str]


DATASETS = {
    "sst": "http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip",
    "mr": "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz",
}


class TextDataset(Dataset):
    def __init__(self, dataset: List[Example], attributes: dict = None):
        self.dataset = dataset
        self.attributes = attributes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return example.text, example.label
