# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Implementation Summary

This repository contains my implementation of the CS336 Assignment 1
components across four modules under `src/`:

- **`tokenizer.py` / `train_bpe.py`** — BPE tokenizer supporting training,
  encoding, decoding, and special tokens (e.g., `<|endoftext|>`), with
  parallelized pretokenization.

- **`models.py`** — Transformer language model built from scratch:
  `LinearModule`, `Embedding`, `RMSNorm`, `SwiGLU`, `RoPE`,
  `MultiHeadSelfAttention` (causal masking), `TransformerBlock`, and
  `TransformerLM`.

- **`utils.py`** — Utility functions for scaled dot-product attention,
  cross-entropy loss, temperature-softmax, top-p (nucleus) sampling,
  warmup + cosine-decay learning rate scheduling, gradient clipping,
  data loading, and checkpoint save/load.

- **`optimizers.py`** — Custom implementations of SGD (with normalized
  step size) and AdamW (with weight decay and bias-corrected moment
  estimates).

All components are wired to the provided test suite via
`tests/adapters.py`. Ad-hoc scripts under `unitest/` include
`test_run_bpe.py` / `test_run_tokenizer.py` for tokenizer validation,
and `model_infer.py` / `model_train.py` for end-to-end inference
and training.