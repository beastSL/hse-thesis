from enum import Enum

from tokenizers import Tokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch

import pandas as pd

class SpecialTokens(Enum):
    UNKNOWN = "[UNK]"
    PADDING = "[PAD]"
    BEGINNING = "[BOS]"
    END = "[EOS]"

class MyDataset(Dataset):
    def __init__(
        self,
        sentences,
        tokenizer: Tokenizer,
        max_len=64,
    ):
        """
        Loads the training dataset and parses it into separate tokenized training examples.
        No padding should be applied at this stage
        :param file_path: Path to the training data
        :param tokenizer: Trained tokenizer
        :param max_len: Maximum length of source and target sentences for each example:
        if either of the parts contains more tokens, it needs to be filtered.
        """
        self.max_len = max_len
        self.special_tokens = {
            SpecialTokens.UNKNOWN.value: tokenizer.convert_tokens_to_ids(SpecialTokens.UNKNOWN.value),
            SpecialTokens.PADDING.value: tokenizer.convert_tokens_to_ids(SpecialTokens.PADDING.value),
            SpecialTokens.BEGINNING.value: tokenizer.convert_tokens_to_ids(SpecialTokens.BEGINNING.value),
            SpecialTokens.END.value: tokenizer.convert_tokens_to_ids(SpecialTokens.END.value),
        }

        self.tokenized = []
        progress_bar = tqdm(list(sentences), leave=False)
        progress_bar.set_description(f"Tokenizing the sentences")
        for line in progress_bar:
            tokenized = tokenizer.encode(line.strip())
            if len(tokenized) + 2 > max_len:
                # self.tokenized.append(None)
                continue
            self.tokenized.append(
                [self.special_tokens[SpecialTokens.BEGINNING.value]] + 
                tokenized + 
                [self.special_tokens[SpecialTokens.END.value]]
            )
        assert len(self.tokenized) == len(sentences)
        self.unit_test()   

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, i):
        return self.tokenized[i]

    def collate_translation_data(self, batch):
        """
        Given a batch of examples with varying length, collate it into `source` and `target` tensors for the model.
        This method is meant to be used when instantiating the DataLoader class for training and validation datasets in your pipeline.
        """
        source = [torch.Tensor(sample) for sample in batch]
        source = pad_sequence(
            source,
            batch_first=True,
            padding_value=self.special_tokens[SpecialTokens.PADDING.value]
        ).type(torch.LongTensor)

        return source
    
    def unit_test(self):
        PAD_IDX = self.special_tokens[SpecialTokens.PADDING.value]
        input = [
            [2, 17, 3],
            [2, 5, 19, 17, 3],
            [2, 17, 18, 3]
        ]
        source = self.collate_translation_data(input)
        source_true = torch.Tensor([
            [2, 17, 3, PAD_IDX, PAD_IDX],
            [2, 5, 19, 17, 3],
            [2, 17, 18, 3, PAD_IDX]
        ])
        assert torch.all(source == source_true)
