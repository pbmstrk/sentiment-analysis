import torch
from torch.nn.utils.rnn import pad_sequence


from .base import VocabMixin, BaseEncoder, TargetEncodingMixin

class BasicEncoder(
    BaseEncoder,
    VocabMixin,
    TargetEncodingMixin
):

    def __init__(self, return_seq_lengths=False):
        self.return_seq_lengths = return_seq_lengths

    def __call__(self, inputs, targets):

        # allow input to also be list instead of nested list 
        if not all(isinstance(inp, list) for inp in inputs):
            inputs = [inputs]

        inputs = self.encode_inputs(inputs)
        targets = self.encode_targets(targets)

        seq_lengths = torch.tensor([len(t) for t in inputs])

        inputs = pad_sequence(inputs, batch_first=True)

        if self.return_seq_lengths:
            return inputs.long(), targets.long()
        
        return inputs.long(), targets.long(), seq_lengths

    def encode_inputs(self, inputs):
        return [torch.tensor(self.convert_token_to_ids(inp)) for inp in inputs]

    def encode_targets(self, targets):
        return torch.tensor([self.convert_targets(tar) for tar in targets])

    def collate_fn(self, batch):

        unzip_batch = lambda x: list(map(list, zip(*batch))

        inputs, targets = unzip_batch(batch)

        return self(inputs=inputs, targets=targets)



class LSTMEncoder(BaseEncoder):
    def __init__(self, vocab, target_encoding):
        self.vocab = vocab
        self.target_encoding = target_encoding

    def __call__(self, batch):

        batch = [self._encode(item) for item in batch]

        data = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        seqlengths = [len(el) for el in data]

        x = pad_sequence(data, batch_first=True)

        return x.long(), torch.Tensor(targets).long(), seqlengths

    def _encode(self, example):

        text = torch.tensor([self.vocab[word] for word in example[0]])
        label = torch.tensor(self.target_encoding[example[1]])

        return text, label


class TransformerEncoder(BaseEncoder):
    def __init__(self, vocab, target_encoding):
        self.vocab = vocab
        self.target_encoding = target_encoding

        assert hasattr(vocab, "cls_token") and hasattr(vocab, "sep_token")

    def __call__(self, batch):

        batch = [self._encode(item) for item in batch]

        data = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        x = pad_sequence(data, batch_first=True)

        return x.long(), torch.Tensor(targets).long()

    def _encode(self, example):

        text = torch.tensor(
            [self.vocab.cls_token]
            + [self.vocab[word] for word in example[0]]
            + [self.vocab.sep_token]
        )
        label = torch.tensor(self.target_encoding[example[1]])

        return text, label
