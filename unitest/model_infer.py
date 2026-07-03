import sys
import os
import timeit

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # stanford-cs336-assignment1-basics

sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
from models import *
from utils import *
from optimizers import *
from train_bpe import *
from tokenizer import *

MAX_TOKENS = 1000


gstart = timeit.default_timer()

## Initialize model and optimizer
model_hyperparams = {
    "vocab_size": 10000,
    "context_length": 256, 
    "d_model": 512, 
    "num_layers": 4, 
    "num_heads": 16, 
    "d_ff": 1344, 
    "rope_theta": 10000.0,
    "device": device
}
model = TransformerLM(**model_hyperparams).to(device)
optimizer = AdamW(
    model.parameters(), 
    betas = (0.9, 0.999),
    lr = 3e-3,
    weight_decay = 0.1, 
    eps = 1e-8
)

## Load model and set to evaluation mode
checkpoint = load_checkpoint(os.path.join(SCRIPT_DIR, f"./checkpoint/checkpoint_tiny_stories_9999.pt"), model, optimizer)
model.eval()
torch.no_grad()

## tokenizer trained on TinyStory training data
vocab_path = os.path.join(SCRIPT_DIR, "train_bpe/tiny_stories_vocab.json")
merges_path = os.path.join(SCRIPT_DIR, "train_bpe/tiny_stories_merges.txt")
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

prompt = "What is happening"
tokens = tokenizer.encode(prompt)
EOT_ID = tokenizer.encode("<|endoftext|>")[0]
CONTEXT_LENGTH = model_hyperparams["context_length"]
count = 0

while count < MAX_TOKENS:
    if len(tokens) >= CONTEXT_LENGTH:
        model_input = torch.tensor([tokens[-CONTEXT_LENGTH:]]).to(device)
    else:
        model_input = torch.tensor([tokens]).to(device)
    next_token_probs = softmax_temperature(model(model_input)[0, -1, :], dim=-1, temperature=0.5)
    next_token_probs = top_sampling(next_token_probs, threshold=0.6)
    sampled_idx = torch.multinomial(next_token_probs, num_samples=1)
    if sampled_idx.item() == EOT_ID:
        break
    tokens.append(sampled_idx.item())
    count += 1
print(tokenizer.decode(tokens))