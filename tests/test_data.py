from text_classification.datasets import SSTDataset, MRDataset
from text_classification.tokenizers import SpacyTokenizer, SimpleTokenizer


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

    def test_sst(self):

        root = '.testdata'

        # test with default SpacyTokenizer
        train, val, test = SSTDataset(filter_func=lambda x: x.label != 'neutral', root=root,
        tokenizer=SpacyTokenizer())
        assert_size_sst(train, val, test)
        assert train.attributes['tokenizer'] == SpacyTokenizer().__str__()

        # test access
        train[0]

        # test with simple and subtrees
        train, val, test = SSTDataset(filter_func=lambda x: x.label != 'neutral', 
                                root=root, tokenizer=SimpleTokenizer(), train_subtrees=True)
        assert train.attributes['tokenizer'] == SimpleTokenizer().__str__()

    def test_mr(self):

        root = '.testdata'

        # test with default SpacyTokenizer
        mr_data = MRDataset(root=root, tokenizer=SpacyTokenizer())
        assert_size_mr(mr_data)
        assert mr_data.attributes['tokenizer'] == SpacyTokenizer().__str__()

        # test access
        mr_data[0]

        mr_data = MRDataset(root=root, tokenizer=SimpleTokenizer()) 
        assert_size_mr(mr_data)
        assert mr_data.attributes['tokenizer'] == SimpleTokenizer().__str__()
        



        
