# MiniGPT Parameter Count Explained — 15,691,776

**Config:** `d_model=384, num_heads=8, num_layers=8, d_ff=1536, vocab_size=2000, max_seq_len=256`

> All `Linear` layers have `bias=False`. No LayerNorm. No positional embedding.
> These are the post-enhancement settings (easy + medium changes applied).

---

## 1. Token Embedding — 768,000

```
nn.Embedding(vocab_size, d_model)
= 2,000 × 384
= 768,000
```

Every token in the vocabulary gets a vector of size 384.
This is a lookup table with 2,000 rows and 384 columns.

---

## 2. One GPTBlock — 1,769,472

Each block has two parts: Attention and Feed-Forward.

### Attention (W_q, W_k, W_v, W_o) — 589,824

```
Each of W_q, W_k, W_v, W_o is nn.Linear(384, 384, bias=False)
= 384 × 384 = 147,456 params each

4 matrices × 147,456 = 589,824
```

Think of it as 4 square matrices, each 384×384.

### Feed-Forward (linear1, linear2) — 1,179,648

```
linear1: nn.Linear(384, 1536, bias=False) = 384 × 1,536 = 589,824
linear2: nn.Linear(1536, 384, bias=False) = 1,536 × 384 = 589,824

589,824 + 589,824 = 1,179,648
```

The FFN expands to 4× width (384 → 1536) then projects back (1536 → 384).

### Per Block Total

```
589,824  (attention)
+ 1,179,648  (feed-forward)
─────────────
1,769,472
```

---

## 3. All 8 Blocks — 14,155,776

```
8 × 1,769,472 = 14,155,776
```

---

## 4. Output Projection — 768,000

```
nn.Linear(d_model, vocab_size, bias=False)
= 384 × 2,000
= 768,000
```

Maps the final hidden state back to a score for each token in the vocabulary.

---

## Grand Total

```
Token embedding      =    768,000
8 × attention        =  4,718,592   (8 × 589,824)
8 × feed-forward     =  9,437,184   (8 × 1,179,648)
Output projection    =    768,000
─────────────────────────────────
TOTAL                = 15,691,776
```

---

## Why Zero Params from RoPE, RMSNorm, and Softcap?

| Component | Params | Why |
|-----------|--------|-----|
| RoPE (cos/sin buffers) | 0 | Computed from a formula, not learned |
| RMSNorm | 0 | `F.rms_norm` with no scale/shift — pure math |
| Logit softcap | 0 | `15 * tanh(x / 15)` — fixed formula, no weights |
| Dropout | 0 | Only active during training, no weights |

---

## Where Does the Compute Go?

```
Feed-forward:   9,437,184  ──  60.1% of all params
Attention:      4,718,592  ──  30.1%
Embeddings:     1,536,000  ──   9.8%  (token in + token out)
```

The **FFN dominates** — this is true of virtually all transformers.
The 4× expansion in `d_ff` means the feed-forward always holds roughly **2× more params than attention**.

---

## Compared to the Original Model (before enhancements)

| Component | Original | Current | Difference |
|-----------|---------|---------|------------|
| Token embedding | 768,000 | 768,000 | — |
| Positional embedding | 98,304 | 0 | −98,304 (replaced by RoPE) |
| Attention per block | 592,896 | 589,824 | −3,072 (biases removed) |
| Feed-forward per block | 1,183,744 | 1,179,648 | −4,096 (biases removed) |
| LayerNorm per block | 1,536 | 0 | −1,536 (replaced by RMSNorm) |
| Final LayerNorm | 768 | 0 | −768 |
| Output projection | 770,000 | 768,000 | −2,000 (bias removed) |
| **Total** | **15,832,784** | **15,691,776** | **−141,008 (−0.9%)** |
