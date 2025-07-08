import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RhetoricalRoleDataset(Dataset):
    def __init__(self, filepath, word2idx, label2idx):
        self.data = pd.read_parquet(filepath)
        self.samples = []
        self.skipped = 0

        for _, row in self.data.iterrows():
            tokens = row['text']
            labels = row['labels']

            # ✅ Convert from np.ndarray to list if needed
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()

            # ✅ Skip if invalid
            if not isinstance(tokens, list) or not isinstance(labels, list) or len(tokens) != len(labels):
                self.skipped += 1
                continue

            token_ids = [word2idx.get(token.lower(), word2idx["<UNK>"]) for token in tokens]
            label_ids = [label2idx.get(str(label), label2idx["<PAD>"]) for label in labels]

            self.samples.append((token_ids, label_ids, len(token_ids)))

        print(f"✅ Loaded {len(self.samples)} samples. Skipped {self.skipped} due to mismatch.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, labels, length = self.samples[idx]
        return torch.tensor(tokens), torch.tensor(labels), length

def collate_fn(batch):
    token_seqs, label_seqs, lengths = zip(*batch)
    max_len = max(lengths)

    padded_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i in range(len(batch)):
        padded_tokens[i, :lengths[i]] = token_seqs[i]
        padded_labels[i, :lengths[i]] = label_seqs[i]

    return padded_tokens, padded_labels, list(lengths)

def prepare_sequence(seq, word2idx):
    """
    Convert a list of tokens into a tensor of corresponding indices.
    """
    idxs = [word2idx.get(word.lower(), word2idx.get("<UNK>", 0)) for word in seq]
    return torch.tensor(idxs, dtype=torch.long)

import re

def tokenize_text(text):
    """
    Simple whitespace and punctuation-based tokenizer.
    You can replace this with your actual tokenizer logic if needed.
    """
    tokens = re.findall(r'\w+|\S', text)
    return tokens
