"""
Generate token_bytes.pt for a tokenizer that wasn't created by tok_train.py.

Usage:
    NANOCHAT_TOKENIZER_DIR=/path/to/tokenizer python -m scripts.generate_token_bytes
"""
import os
import torch
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()

tokenizer_dir = os.environ.get("NANOCHAT_TOKENIZER_DIR")
if tokenizer_dir is None:
    from nanochat.common import get_base_dir
    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

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
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
torch.save(token_bytes, token_bytes_path)
print(f"Saved token_bytes ({vocab_size} tokens) to {token_bytes_path}")
