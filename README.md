# Building a GPT From Scratch (Character-Level, on Shakespeare)

A minimal, from-scratch implementation of a GPT-style transformer, built
incrementally — bigram baseline → self-attention → multi-head attention →
residual connections → layernorm → feedforward → stacked transformer blocks.
Every component was implemented and understood individually before
assembling the next, rather than copying a complete architecture upfront.

## What's in here

- `bigram.py` — baseline: embedding lookup → direct prediction, no attention
- `v2.py` — full architecture: token + positional embeddings, multi-head
  self-attention, residual connections, layernorm, feedforward, N stacked
  transformer blocks

## Architecture

Data flow through `v2.py` (B = batch, T = positions, C = `n_embd`):

```
characters (B, T)               integer IDs
  → token embedding             (B, T, 32)   which character
  + position embedding          (T, 32)      which position, broadcast over B
  → N x Block                   (B, T, 32)
  → final layernorm
  → linear head                 (B, T, 65)   a score for each possible next char
  → softmax + sample            next character
```

- Token embedding + positional embedding
- N stacked blocks (N = 4), each:
  - Multi-head self-attention (4 heads of size 8) with residual connection + pre-norm
  - Feedforward (4x expansion) with residual connection + pre-norm
- Final layernorm → linear projection to vocabulary
- Causal mask: each position attends only to itself and earlier positions
- Context window: 8 characters (`block_size`)

## Results

Both models were trained for 10,000 steps (AdamW, lr 1e-3, batch of 4
sequences x 8 characters) on CPU. Loss is cross-entropy in nats, averaged
over 500 random batches per split. A model guessing uniformly over the 65
characters would score ln(65) = 4.17.

| Model       | Parameters | Train loss | Val loss | Training time |
| ----------- | ---------- | ---------- | -------- | ------------- |
| `bigram.py` | 4,225      | 2.54       | 2.56     | ~6 s          |
| `v2.py`     | 50,753     | 2.10       | 2.15     | ~80 s         |

The transformer's loss is clearly lower, and train and val are close, so it
is not overfitting. Training is short and the batches are tiny, so the loss
printed during training is noisy from step to step. Numbers will vary a
little between runs because no random seed is set.

Sample output (`v2.py`, 500 characters generated from a newline):

```
Ivall gaend a proorrelty o by, in, oustile, intucke're ver, artwich somenst is semeldsorraalt to say, pauresper?
For would in!

QUENEENE:
Thas go. That sumgent.

ROS:
That of of I my? Krom Hot,
Ow loattell.
```

It picks up the play format (speaker names in capitals followed by a colon,
line breaks, punctuation) but the words are mostly nonsense, as expected at
this size. The bigram baseline produces the same kind of text with much less
structure.

## What this is NOT

A production-scale LLM. This is a character-level model (~65 token
vocabulary) with ~50K parameters trained on ~1MB of text, built purely
to understand the mechanics — not to generate quality text. Real models
differ by orders of magnitude in scale, not architecture.

## Running it

Requires Python 3.10+ (developed on 3.14).

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get the dataset (tiny Shakespeare, ~1MB) into the project folder
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 4. Train and generate
python bigram.py   # baseline model
python v2.py       # full transformer
```

`v2.py` trains for 10,000 steps, printing the loss every 1,000 steps, then
prints 500 generated characters. Both scripts read `input.txt` from the
current directory, so run them from the project root.

Hyperparameters (`n_embd`, `batch_size`, `block_size`, `n_layer`) are
variables at the top of `v2.py`.

`main.py` is a small helper that downloads the dataset and prints its
length and first 200 characters, as a quick check that the download works.

