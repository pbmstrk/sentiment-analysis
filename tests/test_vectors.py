import numpy as np

from text_classification.vectors.base import Vectors
from text_classification.encoders.base import VocabMixin


class TestVectors:
    def test_initialise(self):

        vector_map = {
            "test": np.array([1.0, 1.0, 1.0]),
            "vectors": np.array([2.0, 2.0, 2.0]),
        }

        # initialise
        vec = Vectors(3, vector_map)

        # get matrix from vocab object
        vocab = VocabMixin().add_vocab(["test", "vectors"])
        matrix = vec.get_matrix(vocab.vocab)
        assert all(matrix[vocab.vocab["test"]] == vector_map["test"])

        # get matrix from list
        vocab = ["test", "vectors"]
        matrix = vec.get_matrix(vocab)
        assert all(matrix[vocab.index("test")] == vector_map["test"])
