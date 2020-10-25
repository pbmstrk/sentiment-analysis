import os
from typing import Dict, List, Union

import numpy as np
from tqdm import tqdm

from ..utils import download_extract
from ..vocab import Vocab


def extract_vectors(filepath: str) -> Dict:

    embedding_map = {}
    with open(filepath) as embed_file:
        for line in tqdm(embed_file):
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype="float32")
                embedding_map[word] = coefs
            except ValueError:
                continue
    return embedding_map


def GloVe(name: str, dim: int, root: str = ".data"):

    URLs = {
        "42B": "https://nlp.stanford.edu/data/glove.42B.300d.zip",
        "840B": "https://nlp.stanford.edu/data/glove.840B.300d.zip",
        "twitter.27B": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
        "6B": "https://nlp.stanford.edu/data/glove.6B.zip",
    }

    download_extract(URLs[name], name=name, root=root)

    filename = f"glove.{name}.{dim}d.txt"
    filepath = os.path.join(root, name, filename)

    vector_map = extract_vectors(filepath)

    return Vectors(dim, vector_map)


# def Word2Vec(name: "str" = "GoogleNews", root: str = '.data'):
#
#    URL = {'https://drive.google.com/u/0/open?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM'}

#    download_extract(URL, name=name, root=root)

# perhaps load with gensim and covert to dict
# on pause at the moment


class Vectors:
    def __init__(self, dim: int, vector_map: Dict):
        self.vector_map = vector_map
        self.dim = dim

    def get_matrix(self, vocab: Union[List, Vocab]) -> np.ndarray:

        matrix_len = len(vocab)
        weights_matrix = np.zeros((matrix_len, self.dim))
        for i, word in enumerate(vocab):
            try:
                weights_matrix[i] = self.vector_map[word]
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.25, size=(self.dim,))
        return weights_matrix
