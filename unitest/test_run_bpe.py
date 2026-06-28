import os
import sys
import timeit
import logging

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # stanford-cs336-assignment1-basics

sys.path.insert(0, PROJECT_DIR)

from src.train_bpe import *


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(SCRIPT_DIR, "test_run_bpe.log")),
            logging.StreamHandler(),
        ]
    )
    out_dir = os.path.join(SCRIPT_DIR, "train_bpe")
    os.makedirs(out_dir, exist_ok=True)

    gstart = timeit.default_timer()

    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    input_path = os.path.join(PROJECT_DIR, "data", "TinyStoriesV2-GPT4-train.txt")

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    save_vocab(vocab, os.path.join(out_dir, "tiny_stories_vocab.json"))
    save_merges(merges, os.path.join(out_dir, "tiny_stories_merges.txt"))

    gstop = timeit.default_timer()
    logging.info(f"Total Execution Time of TinyStory: {(gstop - gstart)/60:.2f} minutes")

    gstart = timeit.default_timer()

    special_tokens=["<|endoftext|>"]
    vocab_size = 32000
    input_path = os.path.join(PROJECT_DIR, "data", "owt_train.txt")

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    save_vocab(vocab, os.path.join(out_dir, "owt_vocab.json"))
    save_merges(merges, os.path.join(out_dir, "owt_merges.txt"))

    gstop = timeit.default_timer()
    logging.info(f"Total Execution Time of OWT: {(gstop - gstart)/60:.2f} minutes")