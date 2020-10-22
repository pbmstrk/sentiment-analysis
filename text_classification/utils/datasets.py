from typing import Callable, List, Optional
from nltk.tree import Tree

from ..datasets.base import Example


def parse_line_tree(line, subtrees: bool = True):

    tree = Tree.fromstring(line)

    if subtrees:
        return ([" ".join(t.leaves()), t.label()] for t in tree.subtrees())
    return ([" ".join(tree.leaves()), tree.label()],)


def parse_line_without_label(line, label):

    return ([" ".join(line.split()), label],)


def get_data_from_file(filepath: str, parser: Callable, errors: Optional[str] = None):

    exs = []
    with open(filepath, errors=errors if errors else None) as f:
        for line in f:
            exs.extend(parser(line))
    return exs


def map_list_to_example(
    element: List,
    tokenizer: Optional[Callable],
    filter_func: Optional[Callable],
    label_map: Optional[dict] = None,
):

    ex = Example(
        text=tokenizer(element[0]) if tokenizer else element[0],
        label=label_map[element[1]] if label_map else element[1],
    )

    if filter_func:
        return ex if filter_func(ex) else None
    return ex
