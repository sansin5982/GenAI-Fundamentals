```python
# 🧰 Setup: Import Libraries
import torch
import torch.nn as nn
import math
import numpy as np
```


```python
# 1. Token Embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)  # Shape: [batch_size, seq_len, d_model]
```

* Explanation:

Each token (e.g., "India") becomes a vector of size d_model.


```python
# 2. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
```

Explanation:

Adds "position" awareness so the model knows the order of words.


```python
# 3. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # Linear projection
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.out(output)

```


```python
# 4. Feed Forward + Add & Norm
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        # sublayer_output must already be a Tensor
        return self.norm(x + self.dropout(sublayer_output))


```


```python
# 5. Encoder Block
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, x, x, mask)     # <-- Call attention first
        x = self.addnorm1(x, attn_output)          # <-- Then pass result
        ff_output = self.ff(x)
        x = self.addnorm2(x, ff_output)
        return x

```


```python
# 6. Decoder Block
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)
        self.addnorm3 = AddNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.addnorm1(x, self_attn_output)

        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.addnorm2(x, cross_attn_output)

        ff_output = self.ff(x)
        x = self.addnorm3(x, ff_output)

        return x

```


```python
# 7. Full Transformer Model
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, num_heads, num_layers):
        super().__init__()
        self.src_embed = nn.Sequential(TokenEmbedding(src_vocab, d_model), PositionalEncoding(d_model))
        self.tgt_embed = nn.Sequential(TokenEmbedding(tgt_vocab, d_model), PositionalEncoding(d_model))

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.src_embed(src)
        tgt = self.tgt_embed(tgt)

        for layer in self.encoder:
            src = layer(src, src_mask)

        for layer in self.decoder:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        return self.fc_out(tgt)

```


```python
# 8. Dummy Data (Simulating: “India is great” → “भारत महान है”)

src_vocab_size = 1000
tgt_vocab_size = 1000
d_model = 512
num_heads = 8
num_layers = 2

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers)

# Example tokenized inputs
src_input = torch.tensor([[5, 23, 78]])  # "India is great"
tgt_input = torch.tensor([[1, 89, 67]])  # "<SOS> भारत महान"

output = model(src_input, tgt_input)
print(output.shape)  # [batch_size, tgt_seq_len, tgt_vocab_size]

```

    torch.Size([1, 3, 1000])
    

#### ✅ Summary:
* `src_input`: token IDs for English sentence
* `tgt_input`: token IDs for Hindi sentence (starts with <SOS>)
* `output`: probability distribution over vocabulary for each target position

#### 📦 Output Shape Explanation: [1, 3, 1000]

| Dimension | Meaning                                                                                                           |
| --------- | ----------------------------------------------------------------------------------------------------------------- |
| `1`       | **Batch size** – you're processing 1 sentence                                                                     |
| `3`       | **Target sequence length** – e.g., `भारत महान` (3 tokens)                                                         |
| `1000`    | **Target vocabulary size** – model predicts a probability for **each word in the target vocab** at each time step |



```python

```
