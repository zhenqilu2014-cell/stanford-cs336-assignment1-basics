import os
import sys
import timeit
import logging
import numpy as np

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # stanford-cs336-assignment1-basics
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, "test_run_tokenizer.log")),
        logging.StreamHandler(),
    ]
)

sys.path.insert(0, PROJECT_DIR)

from src.train_bpe import *
from src.tokenizer import *

num_processes = 4

def tokenizer_encode(input_path, tokenizer):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_processes, split_special_token=b"<|endoftext|>")
    initial_length = 0
    encoded = list()
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        with open(input_path, "rb") as f:
            f.seek(s)
            raw = f.read(e - s).decode("utf-8", errors="ignore")
            raw = raw.replace('\r', '')  # normalize Windows line endings for cross-platform compatibility
            initial_length += len(raw)
            encoded += tokenizer.encode(raw)
    return encoded, initial_length

def save_encoded(encoded, encoded_filepath):
    np.save(encoded_filepath, np.array(encoded, dtype=np.uint16))

## tokenizer trained on TinyStory training data
vocab_path = os.path.join(SCRIPT_DIR, "train_bpe/tiny_stories_vocab.json")
merges_path = os.path.join(SCRIPT_DIR, "train_bpe/tiny_stories_merges.txt")
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

gstart = timeit.default_timer()

input_path = os.path.join(PROJECT_DIR, "data/TinyStoriesV2-GPT4-valid.txt")
encoded, initial_length = tokenizer_encode(input_path, tokenizer)
save_encoded(encoded, os.path.join(SCRIPT_DIR, "data/encoded_tiny_stories_valid.npy"))
logging.info(f"Compression ratio of TinyStory Tokenizer on TinyStory validation data is {len(encoded) / initial_length:.2%}.")

input_path = os.path.join(PROJECT_DIR, "data/TinyStoriesV2-GPT4-train.txt")
encoded, initial_length = tokenizer_encode(input_path, tokenizer)
save_encoded(encoded, os.path.join(SCRIPT_DIR, "data/encoded_tiny_stories_train.npy"))
logging.info(f"Compression ratio of TinyStory Tokenizer on TinyStory training data is {len(encoded) / initial_length:.2%}.")

input_path = os.path.join(PROJECT_DIR, "data/owt_valid.txt")
encoded, initial_length = tokenizer_encode(input_path, tokenizer)
logging.info(f"Compression ratio of TinyStory Tokenizer on OWT validation data is {len(encoded) / initial_length:.2%}.")

del tokenizer

## tokenizer trained on OWT training data
vocab_path = os.path.join(SCRIPT_DIR, "train_bpe/owt_vocab.json")
merges_path = os.path.join(SCRIPT_DIR, "train_bpe/owt_merges.txt")
special_tokens = ["<|endoftext|>"]

tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

input_path = os.path.join(PROJECT_DIR, "data/owt_valid.txt")
encoded, initial_length = tokenizer_encode(input_path, tokenizer)
save_encoded(encoded, os.path.join(SCRIPT_DIR, "data/encoded_owt_valid.npy"))
logging.info(f"Compression ratio of OWT Tokenizer on OWT validation data is {len(encoded) / initial_length:.2%}.")

input_path = os.path.join(PROJECT_DIR, "data/owt_train.txt")
encoded, initial_length = tokenizer_encode(input_path, tokenizer)
save_encoded(encoded, os.path.join(SCRIPT_DIR, "data/encoded_owt_train.npy"))
logging.info(f"Compression ratio of OWT Tokenizer on OWT training data is {len(encoded) / initial_length:.2%}.")

del tokenizer

gstop = timeit.default_timer()
logging.info(f"Total Execution Time of OWT: {(gstop - gstart)/60:.2f} minutes")