"""
Generate token_bytes.pt for a pre-trained tokenizer.

token_bytes.pt is normally produced as a side-effect of tok_train.py, but when
using a pre-existing tokenizer this script creates it standalone.

Usage:
    python -m scripts.generate_token_bytes
"""
import os
import torch
from nanochat.tokenizer import get_tokenizer, get_tokenizer_dir

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())

token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)
    else:
        token_bytes.append(len(token_str.encode("utf-8")))

token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")

tokenizer_dir = get_tokenizer_dir()
out_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(out_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes.pt ({vocab_size} tokens) to {out_path}")
