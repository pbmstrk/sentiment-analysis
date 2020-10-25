from text_classification.datasets import MRDataset, SSTDataset
from text_classification.tokenizers import SimpleTokenizer, SpacyTokenizer


def assert_size_sst(train, val, test):
    true_train_size = 6920
    true_val_size = 872
    true_test_size = 1821

    assert len(train) == true_train_size
    assert len(val) == true_val_size
    assert len(test) == true_test_size


def assert_size_mr(data):
    true_data_size = 10662

    assert len(data) == true_data_size


class TestData:
    def test_sst(self, tmpdir):

        # test with default SpacyTokenizer
        train, val, test = SSTDataset(
            filter_func=lambda x: x.label != "neutral",
            root=tmpdir,
            tokenizer=SpacyTokenizer(),
        )
        assert_size_sst(train, val, test)

        # test access
        train[0]

        # test with simple and subtrees
        train, val, test = SSTDataset(
            filter_func=lambda x: x.label != "neutral",
            root=tmpdir,
            tokenizer=SimpleTokenizer(),
            train_subtrees=True,
        )

    def test_mr(self, tmpdir):

        # test with default SpacyTokenizer
        mr_data = MRDataset(root=tmpdir, tokenizer=SpacyTokenizer())
        assert_size_mr(mr_data)

        # test access
        mr_data[0]

        mr_data = MRDataset(root=tmpdir, tokenizer=SimpleTokenizer())
        assert_size_mr(mr_data)
