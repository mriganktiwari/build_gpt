import unicodedata

# few helper functions to help with training tokenizer

def get_stats(ids):
    """
    inp: list of indices
    out: dictionary of pairs of indices and their occurence counts, in descending order
    example: [1,2,3,1,2] -> {(1,2):2, (2,3):1, (3,1):1}
    """
    counts = {}
    for ch1, ch2 in zip(ids, ids[1:]):
        pair = (ch1, ch2)
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    from the stats / counts obtained using get_stats,
    merge and replace a pair with a new integer 'idx'
    'idx' : 'pair' also gets added to the vocab dict
    return: new_ids with 'idx' replaced in all occurences of the 'pair'
    """
    i = 0
    new_ids = []
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class Tokenizer:
    """
    Base class for building text tokenization based on byte-pair encoding
    Inherit this class and build your tokenizer
    """

    def __init__(self):
        # default: vocab_size of 256 (all bytes), no merges, no patterns
        self.merges = {}
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def _build_vocab(self):
        # vocab is simply deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        
        for special, idx in self.special_tokens.items():
            # '<|endoftext|>'.encode('utf-8') --> b'<|endoftext|>'
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose = False):
        """
        train merges and vocab from the given text
        """
        raise NotImplementedError

    def encode(self, text):
        """
        encode the given text
        """
        raise NotImplementedError

    def decode(self, ids):
        """
        ids: list of indices to decode from
        """
        raise NotImplementedError

    def save(self, file_prefix):
        """
        Savens two files: file_prefix.vocab and file_prefix.model
        - model file is the critical one: intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """

        # write the model
        model_file = file_prefix + '.model'
        with open(model_file, 'w') as f:
            # write the version, patters, merges
            f.write("minbpe v1\n")
            f.write(f'{self.pattern}\n')
            # write special tokens, first the number of them, then each one
            f.write(f'{len(self.special_tokens)}\n')
            for special, idx in self.special_tokens.items():
                f.write(f'{special} {idx}]n')
            # write merges dict
            for idx1, idx2 in self.merges:
                # not using the values (new indices) from merged dicts,
                # as they can be progressively incremented fro 256 onwards when reading this file
                f.write(f'{idx1} {idx2}]n')
        
        # write vocab for the human to look at
        vocab_file = file_prefix + '.vocab'
        inverted_merges = {idx:pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w') as f:
            for idx, token in self.vocab.items():
                # many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings
                # we use errors='replace' to replace them with the replacement char �
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f'[{s0}][{s1}] -> [{s}] {idx}\n')
                else:
                    # otherwise this is a leaf token, just print it
                    # this should just be the first 256 tokens, the bytes
                    f.write(f'[{s}] -> {idx}\n')

    def load(self, model_file):
        """
        Inverse of save() but only for the model file
        """
        assert model_file.endswith(".model")

        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding='utf-8') as f:
            # read the version
            version = f.readline().strip()
            assert version == 'minbpe v1'
            # read the patters
            self.pattern = f.readline().strip()
            # read special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
