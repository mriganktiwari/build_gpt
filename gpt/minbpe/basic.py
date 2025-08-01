# basic tokenizer as implemented in the practice
# following gpt2 bpe encoder.py

from base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text processing
        text_bytes = text.encode('utf-8') # get raw bytes
        ids = list(text_bytes) # list of integers in range 0 - 255

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            stats = get_stats(ids)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f'merging {top_pair} into a new token {idx}')
            ids = merge(ids, top_pair, idx)
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

            # print
            if verbose:
                print(f'merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {stats[top_pair]} occurences')
        
        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

def decode(self, ids):
    # given ids (list of integers), return Python string
    text_bytes = b"".join([self.vocab[i] for i in ids])
    text = text_bytes.decode('utf-8', errors='replace')
    return text

def encode(self, text):
    # given a string text, return the token ids
    text_bytes = text.encode("utf-8")
    ids = list(text_bytes)
    while len(ids) >= 2: # <2 len(ids) will need no merge
        stats = get_stats(ids)
        for key in self.merges.keys():
            if key in stats:
                ids = merge(ids, key, self.merges[key])
    return ids
