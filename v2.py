import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
n_embd = 32       # width of every vector inside the model (each character becomes 32 numbers)
batch_size = 4    # B: how many independent sequences we process in parallel
block_size = 8    # T (max): how many characters of context the model can see at once
n_layer = 4       # how many transformer Blocks are stacked


# ---------------------------------------------------------------------------
# One head of self-attention
# Each position looks at itself and EARLIER positions, and takes a weighted
# mix of what it finds there.
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        # Three linear layers, each maps 32 numbers -> head_size numbers:
        self.key = nn.Linear(n_embd,head_size,bias=False)     # "what do I contain?"
        self.query = nn.Linear(n_embd,head_size,bias=False)   # "what am I looking for?"
        self.value = nn.Linear(n_embd,head_size,bias=False)   # "what do I hand over if someone attends to me?"
        # Lower-triangular matrix of 1s (block_size x block_size). Not a learned
        # weight, so it's a buffer: saved with the model and moved to GPU with it.
        # Row t has 1s in columns 0..t = the positions that position t may see.
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        # x: (B, T, C)  = (batch, positions, n_embd)
        B,T,C = x.shape
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        # Attention scores: how well each position's query matches every position's key.
        # (B,T,hs) @ (B,hs,T) -> (B,T,T). Scaling by sqrt(head_size) keeps the numbers
        # small so softmax doesn't collapse to a single spike.
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        # Block the future: wherever the mask is 0 (above the diagonal) put -inf,
        # so softmax turns those entries into exactly 0 probability.
        # [:T,:T] because the input may be shorter than block_size.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Turn each row into probabilities that add up to 1 (dim=-1 = across the row).
        wei = F.softmax(wei,dim=-1)
        v = self.value(x)   # (B, T, head_size)
        # Weighted average of the values: each position gathers info from the
        # positions it is allowed to see.   (B,T,T) @ (B,T,hs) -> (B,T,hs)
        out = wei @ v
        return out


# ---------------------------------------------------------------------------
# FeedForward: a small network applied to each position on its own.
# Attention moves information BETWEEN positions; this "thinks" about the result
# at each position separately.
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # widen 32 -> 128
            nn.ReLU(),                       # non-linearity: negatives become 0
            nn.Linear(4 * n_embd, n_embd),   # shrink back 128 -> 32
        )
    def forward(self, x):
        return self.net(x)   # (B, T, 32) -> (B, T, 32)


# ---------------------------------------------------------------------------
# Multi-head attention: several small heads side by side, results joined.
# Each head can learn to look for a different pattern.
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # ModuleList (not a plain list) so PyTorch tracks the heads' parameters.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # Run x through every head, then glue the outputs side by side along the
        # LAST dimension (the features): num_heads * head_size = n_embd wide again.
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out


# ---------------------------------------------------------------------------
# Transformer Block = communication (attention) + computation (feed-forward)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head   # 32 // 4 = 8, so 4 heads x 8 = 32
        self.sa_head = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)   # normalizes each position's 32 numbers
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # "x + ..." is a residual connection: the block adds its result to the
        # input instead of replacing it, which makes deep stacks easier to train.
        x = x + self.sa_head(self.ln1(x))   # normalize -> attention -> add back
        x = x + self.ffwd(self.ln2(x))      # normalize -> feed-forward -> add back
        return x


# ---------------------------------------------------------------------------
# The full model
# characters -> vectors -> n_layer Blocks -> scores for the next character
# ---------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Lookup: character ID -> 32 numbers   (vocab_size rows)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Lookup: position 0..block_size-1 -> 32 numbers   (block_size rows)
        self.pos_embedding_table = nn.Embedding(block_size,n_embd)
        # n_layer Blocks in a row, each with 4 attention heads
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(n_layer)])
        # Final layer: 32 numbers -> vocab_size scores (one per possible next character)
        self.lm_head = nn.Linear(n_embd,vocab_size)
        self.ln_f = nn.LayerNorm(n_embd)   # final normalization before the output layer

    def forward(self, idx, targets=None):
        # idx: (B, T) integer character IDs.  targets: (B, T) the correct next characters.
        B,T = idx.shape
        tok_embd = self.token_embedding_table(idx)           # (B, T, n_embd)  which character
        pos_emb = self.pos_embedding_table(torch.arange(T))  # (T, n_embd)     which position
        x = tok_embd + pos_emb      # add them: (B,T,32) + (T,32) -> (B,T,32), position reused for every sequence
        x = self.blocks(x)          # (B, T, n_embd)  attention + feed-forward, n_layer times
        x = self.ln_f(x)            # (B, T, n_embd)
        logits = self.lm_head(x)    # (B, T, vocab_size)  raw score for every possible next character
        if targets is None:
            loss = None             # generating: no correct answer to compare to
        else:
            # cross_entropy wants a flat list of predictions: (N, C) and targets (N,)
            B,T,C = logits.shape
            logits = logits.view(B*T,C)     # (B*T, vocab_size)
            targets = targets.view(B*T)     # (B*T,)
            # How surprised the model is by the right answers: -ln(prob given to correct char), averaged
            loss = F.cross_entropy(logits,targets)

        return logits,loss

    def generate(self,idx,max_new_token):
        # idx: (B, T) the text so far (as IDs). Appends max_new_token characters, one at a time.
        for _ in range(max_new_token):
            # The position table only has block_size rows, so show the model at most
            # the last block_size characters (idx itself keeps growing).
            idx_cond = idx[:, -block_size:]
            logits,loss = self(idx_cond)         # (B, T, vocab_size)
            logits = logits[:,-1,:]              # keep only the LAST position: (B, vocab_size)
            probs = F.softmax(logits,dim=1)      # scores -> probabilities (each row sums to 1)
            idx_next = torch.multinomial(probs,num_samples=1)   # random pick, weighted by probs: (B, 1)
            idx = torch.cat((idx,idx_next),dim=1)               # append: (B, T+1)
        return idx


# ---------------------------------------------------------------------------
# Data: read the text and turn characters into integers
# ---------------------------------------------------------------------------
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

vocab =  sorted(list(set(text)))   # every distinct character, sorted
vocab_size =  len(vocab)           # 65 for tinyshakespeare


stoi = {ch:i for i,ch in enumerate(vocab)}   # character -> integer
itos = {i:ch for i,ch in enumerate(vocab)}   # integer -> character

encode = lambda s: [stoi[c] for c in s]               # string -> list of ints
decode = lambda l: ''.join([itos[i] for i in l])      # list of ints -> string


data = torch.tensor(encode(text), dtype=torch.long)   # whole text as one long tensor of IDs
n = int(0.9 * len(data))
train_data = data[:n]    # first 90% for training
val_data = data[n:]      # last 10% held out to check the model on unseen text

def get_batch(split,batch_size,block_size):
    # Returns x: (batch_size, block_size) inputs and y: same shape, the same text
    # shifted one character to the right (y[i] is the answer for x[i]).
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))   # random start positions
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+1+block_size] for i in ix])
    return x, y

xb,yb = get_batch("train",4,8)   # (batch_size, block_size) = (4, 8)

m = BigramLanguageModel(vocab_size)
logits,loss = m(xb,yb)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)   # updates the weights; lr = step size

for step in range(10000):
    xb, yb = get_batch('train', 4, 8)              # fresh random batch each step
    logits, loss = m(xb, yb)                       # forward pass: predictions and loss
    optimizer.zero_grad(set_to_none=True)          # clear gradients from the previous step
    loss.backward()                                # work out how each weight affects the loss
    optimizer.step()                               # nudge every weight to reduce the loss

    if step % 1000 == 0:
        print(f"step {step}: loss {loss.item():.4f}")


# Generate 500 characters, starting from a single 0 (the newline character)
print(decode(m.generate(torch.zeros((1,1),dtype=torch.long),max_new_token=500)[0].tolist()))
