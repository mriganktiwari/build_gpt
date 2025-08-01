# addon to basic tokenizer: with regex pattern splits
# then use every text after split to build tokenizer

from base import Tokenizer, get_stats, merge
import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    
    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int, dictionary of special tokens
            example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split text into list of text chunks as per self.pattern
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text pre-processing
        ids = [list(ch.encode('utf-8')) for ch in text_chunks] # [[], [], ...]

        merges = {}
        vocab = {idx:bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # this step of getting stats differs from BasicTokenizer
            # iteratively have to update 'stats' for all texts in text_chunks list
            stats = {}
            for chunk_ids in ids:
                # update stats by passing it to get_stats
                get_stats(chunk_ids, stats)
            # find the pair with highest count
            top_pair = max(stats, key=stats.get)
            # mint a new token, assign it to the next available id
            idx = 256 + i
            # replace all occurences of pair in each ids elements with idx
            ids = [merge(chunk_ids, top_pair, idx) for chunk_ids in ids]
            # save the merge
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

            # prints
            if verbose:
                print(f'merge {i+1}/{num_merges}: {top_pair}->{idx} ({vocab[idx]}) had {stats[top_pair]} occurences')
        
        # save class vars
        self.merges = merges
        self.vocab = vocab
    
    def register_special_tokens(self, special_tokens):
        # str -> int
        # dictionary
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in special_tokens.items()}
    
    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode('utf-8'))
            else:
                raise ValueError(f'invalid token id: {idx}')
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode('utf-8', errors='replace')
        return text
    
    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. girst convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            for key in self.merges.keys():
                if key in stats:
                    ids = merge(ids, key, self.merges[key])
        return ids
    
    def encode_ordinary(self, text):
        """
        Encoding that ignores any special tokens
        """
        # split text in to chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded seperatedly, the results are concatenated
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode('utf-8') # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special='none_raise'):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behaviour right now as well
        any other behaviour is either annoying or a major footgun
        """

        # decode the user desure w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        else:
            raise ValueError(f'allowed_special = {allowed_special} not understood')

        if not special:
            # if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        # now all the special characters are seperated from the rest of the text
        # all chunks of text are encoded seperately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it seperately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids