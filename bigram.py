import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 16

torch.manual_seed(810)

# Reading input.txt data on which to train and validate
with open("input.txt", "r") as f:
    text = f.read()

# Getting the list of characters in the dataset and its length
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating a mapping for the characters to then create the encoder and decoder to tokenise the dataset
str_int = {ch: i for i, ch in enumerate(chars)}
int_str = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [str_int[i] for i in s]
decode = lambda l: ''.join([int_str[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)

# Splitting into training and validation data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Generating a small batch of training or testing data based on function call
def get_batch(split):
    # Generating inputs x, targets y based on random offsets(positions) in a list of size batch_size
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y


# Implementing a basic bigram model
class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token reads off the logits for the next token directly from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        return logits
