# 🔷 What Is Attention in General?
In any attention mechanism, we compute:

$$
\large Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt d_k})V
$$

But over what subset of the tokens we compute that attention defines whether it’s global or local.

## 🔹 Global Attention
#### 🧠 Layman Explanation:
> Each token can **see and attend to every other token** in the sequence — full context.

🧑‍🏫 Analogy:
A student listens to **everyone’s answer** in the class before writing his own.

#### 📌 Key Characteristics:
* **Full context** — attends to all tokens
* Memory intensive (scales as $O(n^2)$)
* Used in original Transformer
* Can be slow for very long sequences

#### ✅ Example:
In the sentence:

> “India is the seventh-largest country by area.”

→ The word “country” attends to “India”, “seventh-largest”, and even “by area”.


```python
# Q, K, V are [batch, seq_len, d_model]
scores = torch.matmul(Q, K.transpose(-2, -1))  # shape: [B, L, L]
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)
```

## 🔹 Local Attention
#### 🧠 Layman Explanation:
> Each token can only see and attend to **a fixed-size window of nearby tokens** — limited context.

#### 👨‍👩‍👧‍👦 Analogy:
A student listens only to his **neighbors** to avoid being distracted by the whole class.

#### 📌 Key Characteristics:
* Attention is restricted to a window (e.g., ±2 tokens)
* Scales linearly with sequence length: $O(n \cdot w)$ where $w$ is window size
* Efficient for long sequences (e.g., documents, genomes)
* Used in models like Longformer, BigBird

✅ Example:
In a 5-token sequence:

["The", "sky", "is", "blue", "today"]

If local attention window = 3:
* "blue" only attends to ["is", "blue", "today"]
* Can’t attend to "The" or "sky"

#### 🧮 Mathematical View
| Type   | Computation Scope                                           |
| ------ | ----------------------------------------------------------- |
| Global | All tokens: $Q \cdot K^T$ of size $n \times n$              |
| Local  | Only windowed tokens: e.g., for token i, use $i-w$ to $i+w$ |




```python
# Create a local attention mask (1s for nearby, 0s for distant)
# Example: mask[i, j] = 1 if abs(i - j) <= window_size
masked_scores = scores.masked_fill(mask == 0, float('-inf'))
weights = F.softmax(masked_scores, dim=-1)
output = torch.matmul(weights, V)
```

### 📊 Tabular Comparison
| Feature            | Global Attention                      | Local Attention                    |
| ------------------ | ------------------------------------- | ---------------------------------- |
| Visibility         | Attends to all tokens                 | Attends to nearby tokens only      |
| Computational Cost | $O(n^2)$                              | $O(n \cdot w)$, w = window size    |
| Memory Usage       | High                                  | Lower                              |
| Use Cases          | Short to medium-length text, GPT/BERT | Long documents, DNA, audio, vision |
| Example Models     | BERT, GPT, T5                         | Longformer, BigBird, Reformer      |

### 🧠 Summary in 1 Line
> Global Attention = “I see everyone.”
> Local Attention = “I see only my neighbors.”


```python

```
