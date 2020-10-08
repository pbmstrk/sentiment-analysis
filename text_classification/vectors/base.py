import os

import numpy as np
from ..utils import download_extract
from tqdm import tqdm


def extract_vectors(filepath: str):

    embedding_map = {}
    with open(filepath) as embed_file:
        for line in tqdm(embed_file):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embedding_map[word] = coefs
    return embedding_map


def GloVE(name: str, dim: int, root: str = ".data"):

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

    return Vectors(f"GloVe(name={name}, dim={dim})", vector_map)


def Word2Vec(name: "str" = "GoogleNews", root: str = ".data"):

    URL = {"https://drive.google.com/u/0/open?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"}

    download_extract(URL, name=name, root=root)

    # perhaps load with gensim and covert to dict
    # on pause at the moment


class Vectors:
    def __init__(self, name: str, vector_map: dict):
        self.vector_map = vector_map
        self.name = name

    def __str__(self) -> str:
        return self.name
