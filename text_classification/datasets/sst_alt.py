import os
from functools import partial
from typing import Callable, Optional, Tuple

from ..utils import download_extract
from ..utils.datasets import (
    get_data_from_file,
    map_list_to_example,
    parse_line_label_first,
)
from .base import DATASETS, TextDataset


def SSTDatasetAlt(
    root: str = ".data",
    name: str = "sst_alt",
    train_subtrees: bool = False,
    fine_grained: bool = False,
    tokenizer: Optional[Callable] = None,
    override: Optional[bool] = False,
) -> Tuple[TextDataset, TextDataset, TextDataset]:

    # adapted from https://github.com/pytorch/text/blob/master/torchtext/datasets/sst.py#L34-L36
    prefix = "very " if fine_grained else ""
    label_map = {
        "0": prefix + "negative",
        "1": "negative",
        "2": "neutral",
        "3": "positive",
        "4": prefix + "positive",
    }

    # download and extract dataset
    url = DATASETS["sst-alt"]
    download_extract(url, name, root=root, filename="sst_data.zip", override=override)

    # define a parser to format each example
    parser = parse_line_label_first

    if fine_grained:
        base_file_name = "stsa.fine"
    else:
        base_file_name = "stsa.binary"

    # get data from all files using defined parser
    train = get_data_from_file(
        os.path.join(root, name, base_file_name + ".phrases.train")
        if train_subtrees
        else os.path.join(root, name, base_file_name + ".train"),
        parser,
    )
    val = get_data_from_file(os.path.join(root, name, base_file_name + ".dev"), parser)
    test = get_data_from_file(
        os.path.join(root, name, base_file_name + ".test"), parser
    )

    # data: List of lists. Using map function to filter, tokenize and convert to list of Examples
    map_f = partial(
        map_list_to_example,
        tokenizer=tokenizer,
        label_map=label_map,
    )

    return (
        TextDataset([x for x in map(map_f, train) if x]),
        TextDataset([x for x in map(map_f, val) if x]),
        TextDataset([x for x in map(map_f, test) if x]),
    )
