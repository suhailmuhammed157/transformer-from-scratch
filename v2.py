import torch
import torch.nn as nn
import torch.nn.functional as F

n_embd = 32
batch_size = 4
block_size = 8
n_layer = 4



class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):        
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        v = self.value(x)
        out = wei @ v
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)
        
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # create num_heads separate Head instances

    def forward(self, x):
        # run x through every head independently
        # concatenate all their outputs along the last dimension
        # return the result
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd,vocab_size)        
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_embd = self.token_embedding_table(idx)  # (B, T, n_embd)        
        pos_emb = self.pos_embedding_table(torch.arange(T))
        x = tok_embd + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x) 
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)


        return logits,loss

    def generate(self,idx,max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:] 
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx


with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

vocab =  sorted(list(set(text)))
vocab_size =  len(vocab)


stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for i,ch in enumerate(vocab)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split,batch_size,block_size):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+1+block_size] for i in ix])
    return x, y

xb,yb = get_batch("train",4,8)

m = BigramLanguageModel(vocab_size)
logits,loss = m(xb,yb)


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for step in range(10000): 
    xb, yb = get_batch('train', 4, 8)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"step {step}: loss {loss.item():.4f}")


print(decode(m.generate(torch.zeros((1,1),dtype=torch.long),max_new_token=500)[0].tolist()))
