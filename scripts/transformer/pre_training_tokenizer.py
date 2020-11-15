from text_classification.encoders.base import BaseEncoder


class TransformerEncoderMLM(BaseEncoder):
    def __init__(self, vocab):
        self.vocab = vocab

        assert hasattr(vocab, "cls_token") and hasattr(vocab, "sep_token")
        assert hasattr(vocab, "mask_token")

    def __call__(self, batch):

        batch = [self._encode(item) for item in batch]

        input_ids = [item[0] for item in batch]
        special_token_mask = [item[1] for item in batch]

        input_ids_pad = pad_sequence(input_ids, batch_first=True)
        special_token_mask_pad = pad_sequence(special_token_mask, batch_first=True)
        masked_indices, labels = mask_tokens(input_ids_pad, special_token_mask_pad)

        return input_ids_pad, special_token_mask_pad, masked_indices, labels

    def _encode(self, example):

        encoding = [self.vocab.cls_token] +
            [self.vocab[word] for word in example[0]] +
                    [self.vocab.sep_token]
        
        special_token_mask = get_special_token_mask(encoding)

        return torch.tensor(encoding), torch.tensor(special_token_mask)

    def mask_tokens(self, inputs, special_token_mask):

        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, 0.15)

        probability_matrix.masked_fill_(special_token_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1

        return masked_indices, labels
    def get_special_token_mask(self, input_ids):

        return list(map(lambda x: 1 if x in [vocab.sep_token, vocab.cls_token] else 0, input_ids))

