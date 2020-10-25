from text_classification.vectors.base import Vectors
from text_classification.vocab import Vocab
import numpy as np

class TestVectors:

    def test_initialise(self):

        vector_map = {
            "test": np.array([1., 1., 1.]),
            "vectors": np.array([2., 2., 2.])
        }

        # initialise
        vec = Vectors(3, vector_map)

        # get matrix from vocab object
        vocab = Vocab(["test", "vectors"])
        matrix = vec.get_matrix(vocab)
        assert all(matrix[vocab["test"]] == vector_map["test"])

        # get matrix from list
        vocab = ["test", "vectors"]
        matrix = vec.get_matrix(vocab)
        assert all(matrix[vocab.index("test")] == vector_map["test"])