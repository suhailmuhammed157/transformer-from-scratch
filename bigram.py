import torch
import torch.nn as nn
import torch.nn.functional as F



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)        
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
            logits,loss = self(idx)
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

def get_batch(split, block_size, batch_size):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+1+block_size] for i in ix])
    return x, y

xb,yb = get_batch("train",4,8)

m = BigramLanguageModel(vocab_size)
logits,loss = m(xb,yb)


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for step in range(10000):  # or however many iterations
    xb, yb = get_batch('train', 4, 8)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

print(decode(m.generate(torch.zeros((1,1),dtype=torch.long),max_new_token=500)[0].tolist()))




