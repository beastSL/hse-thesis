from enum import Enum
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch

import pandas as pd
import numpy as np

class SpecialTokens(Enum):
    UNKNOWN = "[UNK]"
    PADDING = "[PAD]"
    BEGINNING = "[BOS]"
    END = "[EOS]"

class MyDataset(Dataset):
    def __init__(
        self,
        file_path,
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
        self.labels = []
        lines = pd.read_csv(file_path)['text'].values
        labels = pd.read_csv(file_path)['score'].values
        progress_bar = tqdm(list(zip(lines, labels)))
        progress_bar.set_description(f"Read the {file_path} corpus")
        for line, label in progress_bar:
            tokenized = tokenizer.encode(line.strip())
            if len(tokenized) + 2 > max_len:
                continue
            self.tokenized.append(
                [self.special_tokens[SpecialTokens.BEGINNING.value]] + 
                tokenized + 
                [self.special_tokens[SpecialTokens.END.value]]
            )
            self.labels.append(label)
        assert len(self.tokenized) == len(self.labels)
        self.unit_test()
                

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, i):
        return self.tokenized[i], self.labels[i]

    def collate_translation_data(self, batch):
        """
        Given a batch of examples with varying length, collate it into `source` and `target` tensors for the model.
        This method is meant to be used when instantiating the DataLoader class for training and validation datasets in your pipeline.
        """
        targets = torch.Tensor([sample[1] for sample in batch])
        source = [torch.Tensor(sample[0]) for sample in batch]
        source = pad_sequence(
            source,
            batch_first=True,
            padding_value=self.special_tokens[SpecialTokens.PADDING.value]
        ).type(torch.LongTensor)
        return source, targets
    
    def unit_test(self):
        PAD_IDX = self.special_tokens[SpecialTokens.PADDING.value]
        input = [
            ([2, 17, 3], 1),
            ([2, 5, 19, 17, 3], 0),
            ([2, 17, 18, 3], 1)
        ]
        source, targets = self.collate_translation_data(input)
        source_true = torch.Tensor([
            [2, 17, 3, PAD_IDX, PAD_IDX],
            [2, 5, 19, 17, 3],
            [2, 17, 18, 3, PAD_IDX]
        ])
        assert torch.all(source == source_true), targets == torch.Tensor([1, 0, 1])
