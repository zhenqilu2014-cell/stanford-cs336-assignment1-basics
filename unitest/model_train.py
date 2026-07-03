import sys
import os
import argparse
import timeit
import logging
import numpy as np

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # stanford-cs336-assignment1-basics
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, "training.log")),
        logging.StreamHandler(),
    ]
)

sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
from models import *
from utils import *
from optimizers import *

gstart = timeit.default_timer()

## Initialize hyperparameters
parser = argparse.ArgumentParser(prog='Training Transformer LM')
## Transformer LM
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--rope_theta", type=float, default=10000.0)

## AdamW
parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
parser.add_argument("--lr_min", type=float, default=1e-4)
parser.add_argument("--lr_max", type=float, default=3e-3)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--eps", type=float, default=1e-8)

## Training
parser.add_argument("--max_iter", type=int, default=10000)
parser.add_argument("--warmup_iters", type=int, default=300)
parser.add_argument("--cosine_cycle_iters", type=int, default=9900)
parser.add_argument("--checkpoint_iter", type=int, default=500)
parser.add_argument("--log_iter", type=int, default=100)
parser.add_argument("--eval_iter", type=int, default=100)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--eval_batch_count", type=int, default=16)

## Initialize model and optimizer
args = parser.parse_args()
model_hyperparams = {
    "vocab_size": args.vocab_size, 
    "context_length": args.context_length, 
    "d_model": args.d_model, 
    "num_layers": args.num_layers, 
    "num_heads": args.num_heads, 
    "d_ff": args.d_ff, 
    "rope_theta": args.rope_theta,
    "device": device
}
model = TransformerLM(**model_hyperparams).to(device)
optimizer = AdamW(
    model.parameters(), 
    betas = args.betas,
    lr = args.lr_max,
    weight_decay = args.weight_decay, 
    eps = args.eps
)

## Specify training and validation data
train_data_filename = os.path.join(SCRIPT_DIR, "data/encoded_tiny_stories_train.npy")
valid_data_filename = os.path.join(SCRIPT_DIR, "data/encoded_tiny_stories_valid.npy")
# file_dtype = np.load(valid_data_filename, mmap_mode="r").dtype
# train_data = np.memmap(train_data_filename, dtype=file_dtype, mode="r")
# valid_data = np.memmap(valid_data_filename, dtype=file_dtype, mode="r")
train_data = np.load(train_data_filename, mmap_mode="r")
valid_data = np.load(valid_data_filename, mmap_mode="r")

## Quick sanity check
assert train_data.max() < args.vocab_size, f"Train data contains token IDs >= {args.vocab_size}"
assert valid_data.max() < args.vocab_size, f"Valid data contains token IDs >= {args.vocab_size}"
assert train_data.min() >= 0, f"Train data contains token IDs < 0"
assert valid_data.min() >= 0, f"Valid data contains token IDs < 0"

## Model training
for train_step in range(args.max_iter):
    lr_curr = learning_rate_schedule(lr_max=args.lr_max, lr_min=args.lr_min, warmup_iters=args.warmup_iters, cosine_cycle_iters=args.cosine_cycle_iters, current_iter=train_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_curr
    optimizer.zero_grad() ## clear old gradients
    X_train, y_train = data_loading(train_data, args.batch_size, args.context_length, device=device)
    y_pred = model(X_train)
    train_loss = cross_entropy(y_pred, y_train)
    train_loss.backward()
    gradient_clipping(model.parameters(), max_norm=args.grad_clip)
    optimizer.step()

    if train_step > 0 and train_step % args.checkpoint_iter == 0:
        os.makedirs(os.path.join(SCRIPT_DIR, "checkpoint"), exist_ok=True)
        save_checkpoint(model, optimizer, train_step, os.path.join(SCRIPT_DIR, f"./checkpoint/checkpoint_tiny_stories_{train_step}.pt"))
    if train_step % args.eval_iter == 0:
        model.eval()
        with torch.no_grad():
            valid_loss = list()
            for _ in range(args.eval_batch_count):
                X_valid, y_valid = data_loading(valid_data, args.batch_size, args.context_length, device=device)
                y_pred = model(X_valid)
                valid_loss.append(cross_entropy(y_pred, y_valid).item())
            valid_loss = sum(valid_loss) / len(valid_loss)
        logging.info(f"Validation loss at step {train_step}: {valid_loss:.6f}")
        model.train()
    if train_step % args.log_iter == 0:
        logging.info(f"Training loss at step {train_step}: {train_loss.item():.6f}")

## Save final state of model
os.makedirs(os.path.join(SCRIPT_DIR, "checkpoint"), exist_ok=True)
save_checkpoint(model, optimizer, train_step, os.path.join(SCRIPT_DIR, f"./checkpoint/checkpoint_tiny_stories_{train_step}.pt"))

gstop = timeit.default_timer()
logging.info(f"Total Execution Time: {(gstop - gstart)/60:.2f} minutes")