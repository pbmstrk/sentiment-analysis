from text_classification.encoders.base import BaseEncoder, VocabMixin
from torch.nn.utils.rnn import pad_sequence
import torch


class TransformerEncoderMLM(BaseEncoder, VocabMixin):
    def __call__(self, inputs):

        # allow input to also be list instead of nested list
        if not all(isinstance(inp, list) for inp in inputs):
            inputs = [inputs]

        inputs = self.encode_inputs(inputs)

        inputs = pad_sequence(inputs, batch_first=True)
        padding_mask = self.get_padding_mask(inputs)
        special_token_mask = self.get_special_token_mask(inputs)

        masked_inputs, masked_labels = self.mask_tokens(
            inputs, special_token_mask, padding_mask
        )

        return masked_inputs.long(), masked_labels.long()

    def encode_inputs(self, inputs):

        assert hasattr(self, "cls_token_index") and hasattr(self, "sep_token_index")

        return [
            torch.tensor(
                [self.cls_token_index]
                + self.convert_token_to_ids(inp)
                + [self.sep_token_index]
            )
            for inp in inputs
        ]

    def encode_targets(self, targets):
        return torch.tensor(self.convert_targets(targets))

    def mask_tokens(self, inputs, special_token_mask, padding_mask):

        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, 0.15)

        probability_matrix.masked_fill_(special_token_mask, value=0.0)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.mask_token_index

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def get_padding_mask(self, input_ids):

        return torch.tensor(
            [
                list(map(lambda x: 1 if x in [self.pad_token_index] else 0, inp))
                for inp in input_ids
            ]
        )

    def get_special_token_mask(self, input_ids):

        return torch.tensor(
            [
                list(
                    map(
                        lambda x: 1
                        if x in [self.sep_token_index, self.cls_token_index]
                        else 0,
                        inp,
                    )
                )
                for inp in input_ids
            ]
        )

    def collate_fn(self, batch):

        inputs, _ = self.unzip_batch(batch)

        return self(inputs=inputs)