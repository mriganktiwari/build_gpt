import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence

class CharDataset(Dataset):
    """
    return pair of (x,y) pairs
    """
    def __init__(self, text, block: int = 128, train_split: float = 0.9):
        super().__init__()
        self.vocab = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(self.vocab)
        self.block_size = block
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(train_split * len(data))
        self.train = data[:n]
        self.val = data[n:]

        self.buf = self.train

    def encode(self, text):
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, tokens):
        return ''.join(self.itos[token] for token in tokens.tolist())

    def __len__(self):
        return len(self.buf) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.buf[idx     : idx + self.block_size    ]
        y = self.buf[idx + 1 : idx + self.block_size + 1]
        return x, y

    def split(self, which: str):
        self.buf = self.train if which == 'train' else self.val
        return self

# ------------------------------------------------------------------------------------------------
# custom data loader classes
class TextLoader:
    def __init__(self, text, train_split=0.9, device='cpu'):
        self.vocab = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(self.vocab)

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(train_split * len(data))
        self.train = data[:n]
        self.val = data[n:]
    
    def encode(self, text):
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
    
    def decode(self, tokens):
        return ''.join(self.itos[token] for token in tokens.tolist())
    
    def get_batch(self, split: str, batch=64, block=128):
        data = self.train if split == 'train' else self.val
        ix   = torch.randint(0, len(data) - block - 1, (batch,))
        x    = torch.stack([data[i : i + block]     for i in ix])
        y    = torch.stack([data[i + 1 : i + block + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

# class TextLoader:
#     def __init__(self, vocab, stoi, itos, train_data, val_data):
#         self.vocab = vocab
#         self.vocab_size = len(vocab)
#         self.stoi = stoi
#         self.itos = itos
#         self.train_data = train_data
#         self.val_data = val_data
    
#     @classmethod
#     def from_text(cls, text, train_split=0.9):
#         vocab = sorted(list(set(text)))
#         stoi = {ch:i for i,ch in enumerate(vocab)}
#         itos = {i:ch for ch,i in stoi.items()}

#         data = [stoi[ch] for ch in text]
#         n = int(train_split * len(data))
#         train_data = data[:n]
#         val_data = data[n:]
#         return cls(vocab, stoi, itos, train_data, val_data)

#     def encode(self, text):
#         return [self.stoi[ch] for ch in text]

#     def decode(self, tokens):
#         return ''.join([self.itos[token] for token in tokens])

#     def get_batch(self, split, batch_size=8, block_size=32):
#         self.block_size = block_size
#         split_data = self.train_data if split == 'train' else self.val_data
#         ix = torch.randint(0, len(split_data) - block_size, (batch_size,))
#         x = [split_data[i : i + block_size] for i in ix]
#         y = [split_data[i+1 : i + block_size + 1] for i in ix]
#         return torch.tensor(x), torch.tensor(y)