import os
import sys
from typing import BinaryIO
import regex as re
import json
from collections import Counter
from multiprocessing import Pool

# Anchor to project root regardless of CWD
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from tests.common import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"


## function provided by the assignment
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def word2bytes(word):
    "Convert word string to tuple of bytes"
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)


def split_by_special_tokens(text, special_tokens):
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    pattern = re.compile(pattern)
    return pattern.split(text)


def pre_tokenize_chunk(args):
    """Worker: reads a byte range, pre-tokenizes, returns a Counter."""
    filepath, start, end, special_tokens = args
    with open(filepath, "rb") as f:
        f.seek(start)
        raw = f.read(end - start).decode("utf-8", errors="ignore")
        raw = raw.replace('\r', '')  # normalize Windows line endings for cross-platform compatibility
    
    chunks = split_by_special_tokens(raw, special_tokens)

    counts = Counter()
    for c in chunks:
        for m in re.finditer(PAT, c):
            counts[word2bytes(m.group(0))] += 1
    return counts


def get_max_pair(pair_count):
    """Return the most frequent pair and its count."""
    if not pair_count:
        return None, 0
    return max(pair_count.items(), key=lambda x: (x[1], x[0]))  # Sort by count, then lexicographically


def apply_merges(word_bytes, merge):
    merged = merge[0] + merge[1]
    index = 0
    new_word_bytes = list()
    while index < len(word_bytes):
        if index < len(word_bytes) - 1 and (word_bytes[index], word_bytes[index + 1]) == merge:
            new_word_bytes.append(merged)
            index += 2
        else:
            new_word_bytes.append(word_bytes[index])
            index += 1
    return tuple(new_word_bytes)


def update_count(word_count, pair_count, merge_pair):
    """Update word_count and pair_count in-place after a BPE merge.

    Only updates pairs that actually change (the merge pair + immediate neighbors).
    Pairs far from the merge site are untouched.
    """
    a, b = merge_pair
    merged = a + b

    # Iterate over a snapshot since we mutate word_count
    for word, freq in list(word_count.items()):
        # --- Find all positions of merge_pair in the old word ---
        positions = []
        for i in range(len(word) - 1):
            if word[i] == a and word[i + 1] == b:
                positions.append(i)

        if not positions:
            continue

        # --- Collect all old pair positions that will be removed ---
        # Each merge at pos kills pairs at (pos-1,pos), (pos,pos+1), (pos+1,pos+2)
        old_removed = set()
        for pos in positions:
            if pos > 0:
                old_removed.add(pos - 1)
            old_removed.add(pos)
            if pos + 2 < len(word):
                old_removed.add(pos + 1)

        # Decrement each affected old pair exactly once
        for i in old_removed:
            p = (word[i], word[i + 1])
            pair_count[p] -= freq
            if pair_count[p] <= 0:
                del pair_count[p]

        # --- Build the merged word ---
        parts = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                parts.append(merged)
                i += 2
            else:
                parts.append(word[i])
                i += 1
        new_word = tuple(parts)

        # --- Collect all new pair positions to add ---
        # Each merge at old_pos lands at new_idx in the new word
        new_added = set()
        for pos in positions:
            new_idx = pos - sum(1 for p in positions if p < pos)
            if new_idx > 0:
                new_added.add(new_idx - 1)
            if new_idx + 1 < len(new_word):
                new_added.add(new_idx)

        # Increment each new pair exactly once
        for i in new_added:
            p = (new_word[i], new_word[i + 1])
            pair_count[p] += freq

        # --- Update word_count in-place ---
        del word_count[word]
        word_count[new_word] += freq

    return word_count, pair_count


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    num_processes = 8
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_processes, split_special_token=b"<|endoftext|>")

    tasks = [(input_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        results = pool.map(pre_tokenize_chunk, tasks)

    # Step 4: aggregate counts
    word_count = Counter()
    for r in results:
        word_count.update(r)
    
    pair_count = Counter()
    for word, freq in word_count.items():
        for pair in zip(word[:-1], word[1:]):
            pair_count[pair] += freq

    base_vocab_size = len(vocab)
    merges = list()
    for vocab_index in range(base_vocab_size, vocab_size):
        # Find the most frequent pair
        pair_select, _ = get_max_pair(pair_count)
        if not pair_select:
            break
        word_count, pair_count = update_count(word_count, pair_count, pair_select)
        vocab[vocab_index] = b"".join(pair_select)
        merges.append(pair_select)

    return vocab, merges

def bytes_to_unicode(b: bytes) -> str:
    byte_encoder = gpt2_bytes_to_unicode()
    return "".join(byte_encoder[byte] for byte in b)

def save_vocab(vocab, vocab_filepath):
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        vocab_json[bytes_to_unicode(token_bytes)] = token_id

    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

def save_merges(merges, merges_filepath):
    with open(merges_filepath, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{bytes_to_unicode(a)} {bytes_to_unicode(b)}\n")