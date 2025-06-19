# Decoder
 responsible for generating output tokens (e.g., translating “India is great” → “भारत महान है”).

 ## 🔷 Transformer Decoder: Overview
The decoder block has three main sub-layers:

1. Masked Multi-Head Self-Attention
2. Cross-Attention (Encoder-Decoder Attention)
3. Feed Forward Network (FFN)

Each sub-layer has:

* **Residual Connection**
* **Layer Normalization**

Let’s break these down with explanations and examples.

## 🔶 Step 1: Masked Multi-Head Self-Attention
#### 🧠 Purpose:
Allows the decoder to **attend to previous output tokens**, but not future ones.

>❗ During training, the decoder should not peek at future words!

For example, when generating Hindi translation:'

Input so far: "भारत"

Can attend to: ["भारत"]

Cannot attend to: ["महान", "है"]


#### 🛠️ How It Works:
Same as regular self-attention, but with a mask that blocks attention to future positions (upper triangle of the matrix).

#### 🧮 Masked Attention (Simplified Python)


```python
import numpy as np

def masked_softmax(scores):
    mask = np.triu(np.ones(scores.shape), k=1)  # upper triangle mask
    scores = np.where(mask == 1, -1e9, scores)  # set to large negative
    exps = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)
```

This ensures:

At position $t$, you can only attend to positions $≤ t$.

## 🔶 Step 2: Cross-Attention (Encoder-Decoder Attention)
### 🧠 Purpose:
Allows the decoder to look at the **encoder output** — the source sentence "India is great".

> ❗ This is how the decoder knows what it’s translating from.

#### 🛠️ How It Works:
* **Query (Q)** → from decoder (masked self-attention output)
* **Key (K), Value (V)** → from encoder output

So, decoder learns:

> "While predicting the next Hindi word, where should I focus in the English sentence?"

Q = decoder hidden state: "महान"

K,V = encoder outputs: "India", "is", "great"

→ focuses attention on "great"

This works like a standard attention mechanism — but between **target and source**.

## 🔶 Step 3: Feed Forward Network (FFN)
Same as in the encoder:

$$
\large FFN(x) = ReLU(xW1 + b1)W2 + b2
$$

It operates on each position separately, with two linear layers and ReLU.

## 🔁 Add & Norm in Decoder
Each of the three main components is followed by:
* **A residual connection**
* **A layer normalization**

So the flow becomes:


```python
Masked Self-Attention
→ Add & Norm
→ Cross Attention
→ Add & Norm
→ FFN
→ Add & Norm

```

### 🔁 Putting It All Together (Decoder Block)



```python
Target Input (Shifted)  → Embedding + Position Encoding
           ↓
  Masked Multi-Head Self-Attention (with mask)
           ↓
        Add & Norm
           ↓
     Cross-Attention (Q from decoder, K,V from encoder)
           ↓
        Add & Norm
           ↓
      Feed Forward Network (FFN)
           ↓
        Add & Norm
           ↓
     Output of Decoder Block
```

#### 🔄 Summary Table
| Layer                       | Purpose                          |
| --------------------------- | -------------------------------- |
| Masked Multi-Head Attention | Attend to previous tokens only   |
| Cross-Attention             | Attend to encoder outputs        |
| FFN                         | Non-linear transformation        |
| Add & Norm (x3)             | Stabilize & retain residual info |


#### ✅ Step 0: Setup Dummy Inputs


```python
import numpy as np

# Dummy target input embeddings (e.g., Hindi tokens)
# Let's say "भारत", "महान", "है" (just proxy vectors)
decoder_input = np.array([
    [0.1, 0.2, 0.3, 0.4],  # भारत
    [0.5, 0.6, 0.7, 0.8],  # महान
    [0.9, 1.0, 1.1, 1.2]   # है
])

# Dummy encoder output (from previous encoder block, e.g., from "India is great")
encoder_output = np.array([
    [1.0, 1.1, 1.2, 1.3],
    [1.4, 1.5, 1.6, 1.7],
    [1.8, 1.9, 2.0, 2.1]
])
```

#### ✅ Step 1: Masked Multi-Head Self-Attention
We simulate one attention head (multi-head is just repeating this in parallel).


```python
def softmax(x, mask=False):
    if mask:
        upper_triangle = np.triu(np.ones(x.shape), k=1)
        x = np.where(upper_triangle == 1, -1e9, x)
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def attention(Q, K, V, mask=False):
    scores = Q @ K.T
    if mask:
        scores = np.where(np.triu(np.ones_like(scores), k=1), -1e9, scores)
    weights = softmax(scores)
    return weights @ V
```

Now, apply it with dummy weights:


```python
# Define weight matrices for self-attention
np.random.seed(42)
W_q = np.random.rand(4, 4)
W_k = np.random.rand(4, 4)
W_v = np.random.rand(4, 4)

# Q, K, V from decoder_input itself
Q = decoder_input @ W_q
K = decoder_input @ W_k
V = decoder_input @ W_v

# Apply masked self-attention
masked_attn_output = attention(Q, K, V, mask=True)
```

#### ✅ Step 2: Add & Norm (after masked self-attention)


```python
def layer_norm(x):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-6)

# Residual connection + LayerNorm
attn1_output = layer_norm(masked_attn_output + decoder_input)

```

#### ✅ Step 3: Cross-Attention (Encoder-Decoder)


```python
# New weights for cross-attention
W_q_cross = np.random.rand(4, 4)
W_k_cross = np.random.rand(4, 4)
W_v_cross = np.random.rand(4, 4)

# Q from decoder (after self-attention), K/V from encoder output
Q_cross = attn1_output @ W_q_cross
K_cross = encoder_output @ W_k_cross
V_cross = encoder_output @ W_v_cross

# Apply cross-attention (no masking needed)
cross_attn_output = attention(Q_cross, K_cross, V_cross)

```

#### ✅ Step 4: Add & Norm (after cross-attention)


```python
# Residual connection + LayerNorm
attn2_output = layer_norm(cross_attn_output + attn1_output)

```

#### ✅ Step 5: Feed Forward Network (FFN)


```python
# Define FFN weights
W1 = np.random.rand(4, 8)
b1 = np.random.rand(8)
W2 = np.random.rand(8, 4)
b2 = np.random.rand(4)

# Apply FFN
hidden = np.maximum(0, attn2_output @ W1 + b1)  # ReLU
ffn_output = hidden @ W2 + b2

```

#### ✅ Step 6: Final Add & Norm (after FFN)


```python
# Residual connection + LayerNorm
decoder_output = layer_norm(ffn_output + attn2_output)

```

#### ✅ Final Decoder Block Output


```python
print("Final Decoder Output:\n", decoder_output)
```

    Final Decoder Output:
     [[-1.38105652  0.32562049 -0.3177617   1.37319773]
     [-1.55247148  0.42240549 -0.05660705  1.18667304]
     [-1.60418732  0.45399003  0.04642535  1.10377194]]
    

#### 🧠 Summary of the Code Flow
* decoder_input → masked self-attention → attn1_output
* attn1_output + encoder_output → cross-attention → attn2_output
* attn2_output → FFN → decoder_output


```python

```
