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

        inputs, targets = self.unzip_batch(batch)

        return self(inputs=inputs, targets=targets)


class TransformerEncoder(
    BaseEncoder,
    VocabMixin,
    TargetEncodingMixin
):

    def __call__(self, inputs, targets):

        # allow input to also be list instead of nested list 
        if not all(isinstance(inp, list) for inp in inputs):
            inputs = [inputs]

        inputs = self.encode_inputs(inputs)
        targets = self.encode_targets(targets)

        inputs = pad_sequence(inputs, batch_first=True)

        return inputs.long(), targets.long(), seq_lengths

    def encode_inputs(self, inputs):

        assert hasattr(self, "cls_token_index") and hasattr(self, "sep_token_index")

        return [torch.tensor(
            [self.cls_token_index]
            + self.convert_token_to_ids(inp)
            + [self.sep_token_index]
        ) for inp in inputs]

    def encode_targets(self, targets):
        return torch.tensor([self.convert_targets(tar) for tar in targets])

    def collate_fn(self, batch):

        inputs, targets = self.unzip_batch(batch)

        return self(inputs=inputs, targets=targets)
