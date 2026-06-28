import os
import sys
from typing import BinaryIO, Iterable
import regex as re
import timeit
import json
from collections import Counter
from multiprocessing import Pool
from train_bpe import *
from tests.common import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None
    ):
        self.vocab = vocab
        self.vocab_reversed = {v: k for k, v in vocab.items()}
        self.merge_rank = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens or []
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] = None
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = {
                int(token_index): bytes(gpt2_byte_decoder[c] for c in token_str)
                for token_str, token_index in json.load(f).items()
            }
        
        if special_tokens:
            for token in special_tokens:
                bytes_token = bytes(gpt2_byte_decoder[c] for c in token)
                if bytes_token not in set(vocab.values()):
                    vocab[len(vocab)] = bytes_token

        gpt2_bpe_merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        
        tokenizer = cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
        return tokenizer
    
    def _merge_word(self, word_bytes: tuple[bytes, ...]) -> list[bytes]:
        """Apply BPE merges in priority order to a single word."""
        tokens = list(word_bytes)

        while True:
            best_index = -1
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_rank and self.merge_rank[pair] < best_rank:
                    best_rank = self.merge_rank[pair]
                    best_index = i
            if best_index == -1:
                break
            merged = tokens[best_index] + tokens[best_index + 1]
            tokens = tokens[:best_index] + [merged] + tokens[best_index + 2:]
        
        return tokens

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        token_ids = list()
        if self.special_tokens:
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
            splitter = re.compile(f"({special_pattern})")

            for part in splitter.split(text):
                if part in self.special_tokens: # special token
                    token_bytes = part.encode("utf-8")
                    token_ids.append(self.vocab_reversed[token_bytes])
                else:
                    for m in re.finditer(PAT, part):
                        pretoken = m.group(0)
                        word_bytes = tuple([bytes([b]) for b in pretoken.encode("utf-8")])
                        token_ids += [self.vocab_reversed[x] for x in self._merge_word(word_bytes)]
        else:
            for m in re.finditer(PAT, text):
                pretoken = m.group(0)
                word_bytes = tuple([bytes([b]) for b in pretoken.encode("utf-8")])
                token_ids += [self.vocab_reversed[x] for x in self._merge_word(word_bytes)]

        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        """
        for line in iterable:
            for token_id in self.encode(line):
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        """Convert a sequence of token IDs back to text."""
        # --- Step 1: token IDs → bytes ---
        token_bytes = []
        for token_id in ids:
            token_bytes.append(self.vocab[token_id])  # each is a bytes object

        # --- Step 2: concatenate all bytes ---
        raw_bytes = b"".join(token_bytes)

        # --- Step 3: bytes → string ---
        text = raw_bytes.decode("utf-8", errors="replace")

        return text