#!/usr/bin/env python3
import torch
import torch.nn as nn
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets):
        logits = self.token_embedding_table(idx)
        return logits

def main():
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print("length of dataset in characters: ", len(text))
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(vocab_size)

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    print(data[:10])

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    block_size = 8
    x = train_data[:block_size]
    y = train_data[1:block_size+1]

    for i in range(block_size):
        context = x[:i+1]
        target = y[i]
        print(f"when input is {context} the target: {target}")

    torch.manual_seed(1337)
    batch_size = 4 # how many independent sequences will we process in parallel?
    block_size = 8 # what is the maximum context length for predictions?

    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    xb, yb = get_batch('train')

    for b in range(batch_size): # batch dimension
        for t in range(block_size): # time dimension
            context = xb[b, :t+1]
            target = yb[b,t]
            print(f"when input is {context.tolist()} the target: {target}")

    m = BigramLanguageModel(vocab_size)
    out = m(xb,yb)
    print(out)


if __name__ == "__main__":
    main()
