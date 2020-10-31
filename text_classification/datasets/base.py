from typing import Callable, Dict, List, NamedTuple, Optional, Union

from torch.utils.data import Dataset

from text_classification.utils.datasets import get_data_from_file, map_list_to_example


# could use Dataclass here (choose not to for python 3.5 compatability)
class Example(NamedTuple):
    text: Union[str, List[str]]
    label: Union[int, str]


DATASETS = {
    "sst": "http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip",
    "mr": "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz",
}


class TextDataset(Dataset):

    r"""
    Base class for Datasets. Subclasses torch.utils.data.Dataset

    Args:
        dataset: Dataset stored in list. Each element of the list is
            a named-tuple with elements text and label.
    """

    def __init__(self, dataset: List[Example]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return example.text, example.label
