from sentiment_analysis.data import SSTDataModuleMLP
import numpy as np


class TestData:
    def test_encoding(self):

        ds = SSTDataModuleMLP()
        train_data, _, _ = ds._download_data(train_subtrees=False)

        ds.setup(min_freq=1, train_subtrees=False)
        encoded_data = ds.train_dataloader().dataset
        inv_enc = {v: k for k, v in ds.encoding.items()}

        ind = np.random.randint(0, len(train_data))

        outputs = []
        for key in encoded_data[ind][0]:
            outputs.append(inv_enc[key.item()])

        assert outputs[1:-1] == train_data[ind].text
