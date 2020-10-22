from typing import List, NamedTuple, Union

from torch.utils.data import Dataset


# could use Dataclass here (choose not to for python 3.5 compatability)
class Example(NamedTuple):
    text: Union[str, List[str]]
    label: Union[int, str]


DATASETS = {
    "sst": "http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip",
    "mr": "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz",
}


class TextDataset(Dataset):
    def __init__(self, dataset: List[Example]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return example.text, example.label
