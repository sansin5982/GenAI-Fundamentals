## 🔹 What is Self-Attention?
> **Self-attention** allows each word in a sentence to **"look at" every other word**, including itself, and decide how much attention to pay to each one.

This is how Transformers **capture context**, relationships, and dependencies between words — no matter how far apart they are.

### 🧠 Real-Life Analogy
Suppose you're reading:

> “The **cat** sat on the **mat**.”

To understand "cat", your brain automatically links it to "sat" (action) and "mat" (location).

**Self-attention mimics this behavior**: for each word, it **calculates attention scores** to every other word in the sentence.

#### 🔧 Step-by-Step: Self-Attention Computation
We begin with input vectors for each word, and we transform them into:
* **Query (Q)**: What am I looking for?
* **Key (K)**: What do I offer?
* **Value (V)**: What information do I hold?

These are obtained by **linear projections** (using learnable weight matrices):

$$
\large Q = XW^Q, K = XW^K, V=XW^V
$$

where:

* X: Input matrix (e.g., embeddings for words)
* $W^Q, W^K, W^V$: Weight matrices

### ⚙ Attention Score Formula
The Scaled Dot-Product Attention for each token is:

$$
\large Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

Breakdown:
| Step              | What it does                                                 |
| ----------------- | ------------------------------------------------------------ |
| $QK^T$            | Measures how similar each query is to each key (dot product) |
| $\sqrt{d_k}$      | Scaling to prevent large values (for stability)              |
| Softmax           | Converts similarity scores into **attention weights**        |
| Multiply with $V$ | Combines information from all words, weighted by importance  |

#### 🔢 Simple Example
Say you have 3 tokens: `"India is great"`

Each token gets Q, K, and V vectors.

Now for the word `"is"`:

* Compute dot product of Q_is with all K_i to get **attention scores**.

* Apply softmax to normalize.

* Multiply those scores with the corresponding V vectors → **context vector**.

This context vector is the **output for "is"**, enriched with information from **all other words** in the sentence.

#### From the previous step (embedding + position encoding):
| Token | Final Vector                     |
| ----- | -------------------------------- |
| India | \[0.1, 1.3, 0.4, 1.5]            |
| is    | \[1.041, 0.640, 0.60999, 1.2999] |
| great | \[1.309, -0.116, 0.92, 1.0998]   |


### Matrix format

X = [
 [0.1, 1.3, 0.4, 1.5],         
 [1.041, 0.640, 0.60999, 1.2999],   
 [1.309, -0.116, 0.92, 1.0998]  
]

#### 🧮 Step-by-Step Calculation (Simple!)
We'll use small 4×4 weight matrices and proxy values for:

* W_Q (Query weight)
* W_K (Key weight)
* W_V (Value weight)



```python
import numpy as np

# Input matrix (X) – word embeddings with position info
X = np.array([
    [0.1, 1.3, 0.4, 1.5],           # India
    [1.041, 0.640, 0.60999, 1.2999],# is
    [1.309, -0.116, 0.92, 1.0998]   # great
])

# Define simple proxy weights for Q, K, V
W_Q = np.array([
    [0.1, 0.2, 0.0, 0.3],
    [0.4, 0.1, 0.2, 0.2],
    [0.3, 0.3, 0.3, 0.0],
    [0.2, 0.0, 0.1, 0.4]
])

W_K = np.array([
    [0.3, 0.1, 0.1, 0.0],
    [0.1, 0.3, 0.2, 0.2],
    [0.2, 0.2, 0.4, 0.1],
    [0.0, 0.4, 0.0, 0.3]
])

W_V = np.array([
    [0.2, 0.1, 0.0, 0.2],
    [0.3, 0.2, 0.3, 0.1],
    [0.0, 0.4, 0.2, 0.2],
    [0.1, 0.0, 0.4, 0.3]
])

# Step 1: Linear projections
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("Query Q:\n", Q)
print("Key   K:\n", K)
print("Value V:\n", V)

```

    Query Q:
     [[0.95     0.27     0.53     0.89    ]
     [0.803077 0.455197 0.440987 0.96026 ]
     [0.58046  0.5262   0.36278  0.80942 ]]
    Key   K:
     [[0.24     1.08     0.43     0.75    ]
     [0.498298 0.938058 0.476096 0.578969]
     [0.5651   0.72002  0.4757   0.39874 ]]
    Value V:
     [[0.56     0.43     1.07     0.68    ]
     [0.53019  0.476096 0.833958 0.784168]
     [0.33698  0.4757   0.58912  0.76414 ]]
    

#### 📊 Step 2: Attention Scores
We compute the attention score between each Query and every Key using dot product:


```python
# Step 2: Attention scores = Q . K.T
scores = Q @ K.T
print("Raw Scores:\n", scores)
```

    Raw Scores:
     [[1.415      1.49427205 1.33825   ]
     [1.59417065 1.59308577 1.37424134]
     [1.4706668  1.42419537 1.20221505]]
    

#### 🧮 Step 3: Scale & Softmax


```python
# Step 3: Scale and softmax
scaled_scores = scores / np.sqrt(4)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

attention_weights = softmax(scaled_scores)
print("Attention Weights:\n", attention_weights)
```

    Attention Weights:
     [[0.33302429 0.34648913 0.32048658]
     [0.34538455 0.34519725 0.3094182 ]
     [0.35070188 0.34264701 0.30665111]]
    

#### 🧮 Step 4: Final Output (Weighted Sum of V)
Now, multiply attention weights with V:


```python
# Step 4: Final self-attention output
output = attention_weights @ V
print("Self-Attention Output:\n", output)

```

    Self-Attention Output:
     [[0.47819624 0.460618   0.83409842 0.74305882]
     [0.48070322 0.46005262 0.83972593 0.74199295]
     [0.48139636 0.45980861 0.84165853 0.74149448]]
    

#### 🎯 Final Output (Interpretation)
* Each row in the output corresponds to a word (“India”, “is”, “great”)
* Each output vector is a mixture of the Value vectors, weighted by how much attention it gives to each other word.
* This enriched vector is sent to the next layer in the Transformer.

#### 🧠 Summary Table
| Step            | What it does                           |
| --------------- | -------------------------------------- |
| Word → Q, K, V  | Linear projection to Query, Key, Value |
| Q ⋅ Kᵀ          | How much each word attends to another  |
| Scale + Softmax | Normalize attention                    |
| Attention × V   | Combine information using attention    |



```python
import torch
import torch.nn.functional as F

# 3 tokens, embedding size 4
X = torch.randn(3, 4)

# Linear layers (weights)
W_Q = torch.randn(4, 4)
W_K = torch.randn(4, 4)
W_V = torch.randn(4, 4)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Attention computation
dk = Q.size(-1) ** 0.5
scores = Q @ K.T / dk
weights = F.softmax(scores, dim=-1)
output = weights @ V

print("Attention Output:\n", output)

```

    Attention Output:
     tensor([[-0.5515,  0.2042, -1.8567,  2.2267],
            [-0.1856, -1.4629, -0.3635,  0.9748],
            [-0.6597, -1.2653, -1.7369,  1.5093]])
    

### 🔁 Multi-Head Attention

Before this, we learned:

* Self-attention calculates how words relate to each other
* It produces a contextualized vector for each word (e.g., for “India”, it sees how “is” and “great” influence it)

### 🧠 Problem with Single Self-Attention:
Single-head attention looks at relationships using only **one "perspective"** or one set of Q, K, V matrices.

👉 But language has multiple patterns:

* Syntax (grammar)
* Semantics (meaning)
* Focus (importance)

### 🧠 Solution: Use Multiple Heads in Parallel

#### 🛠️ Multi-Head Attention = Multiple Self-Attention Mechanisms
Instead of one attention mechanism, we run h parallel attention heads, each with:
* its own weight matrices for Q, K, V
* then **concatenate** all outputs and project them back to the original size

> Instead of doing this process once, Transformers do it **multiple times in parallel** — each with **different learned weight matrices**.

$$
\large head_i = Attention(XW_i^Q, XW_i^K, XW_i^V)
$$

$$
\large MultiHead(X)= Concat(head_1, .... , head_h)W^O
$$

Each **head** captures different types of relationships:
* One may focus on **subject-verb**
* Another may capture **negations**
* Another may detect **semantic similarity**

where:

* $W_i^Q, W_i^K, W_i^V \epsilon R^{d_model \space x \space d_k}$
* $W^O \epsilon R^{hd_k \space x \space d_model}$


After all heads complete their attention:
* Their outputs are concatenated
* Passed through a final linear projection

#### 🔍 Let’s Simplify with an Example:
Assume:

d_model = 4

h = 2 heads

d_k = d_v = 2 (each head works in smaller dimensions)


```python
import numpy as np

# Input embedding (after position encoding)
X = np.array([
    [0.1, 1.3, 0.4, 1.5],        # India
    [1.041, 0.640, 0.60999, 1.2999], # is
    [1.309, -0.116, 0.92, 1.0998]    # great
])

# Define 2 heads → Each with its own Q, K, V
# We'll use small random weights for simplicity
np.random.seed(1)

# head 1
Wq1, Wk1, Wv1 = np.random.rand(4,2), np.random.rand(4,2), np.random.rand(4,2)
# head 2
Wq2, Wk2, Wv2 = np.random.rand(4,2), np.random.rand(4,2), np.random.rand(4,2)

# Output projection W_O (from concat of heads → back to model dim)
W_O = np.random.rand(4, 4)

# Define attention function
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def attention(Q, K, V):
    scores = Q @ K.T
    scaled = scores / np.sqrt(Q.shape[-1])
    weights = softmax(scaled)
    return weights @ V

# Head 1
Q1, K1, V1 = X @ Wq1, X @ Wk1, X @ Wv1
head1 = attention(Q1, K1, V1)

# Head 2
Q2, K2, V2 = X @ Wq2, X @ Wk2, X @ Wv2
head2 = attention(Q2, K2, V2)

# Concatenate heads (axis=1 since each head returns [3 x 2])
concat_heads = np.concatenate([head1, head2], axis=1)

# Final projection
multi_head_output = concat_heads @ W_O

print("Multi-Head Attention Output:\n", multi_head_output)
```

    Multi-Head Attention Output:
     [[2.31513935 1.7957773  3.62141732 3.34509988]
     [2.34133771 1.80480821 3.65242414 3.37862218]
     [2.34870176 1.80751799 3.66099707 3.38815117]]
    

#### 📊 Output:
Each row corresponds to a word (India, is, great), enriched by 2 heads of self-attention that looked at the context from different angles.

### 🎯 Why Multi-Head?
| Feature               | Benefit                                       |
| --------------------- | --------------------------------------------- |
| Multiple perspectives | Understand complex relationships              |
| Parallel computation  | Faster and richer learning                    |
| Flexibility           | Different heads specialize in different tasks |
| Parallel heads           | Capture diverse patterns (syntax, semantics, etc.)      |
| Dimensionality reduction | Each head works in smaller space (e.g., 2 instead of 4) |
| Concatenation            | Combines multiple learned relations                     |
| Output projection        | Brings back to model dimension                          |



#### 📌 Summary
| Component     | Role                           |
| ------------- | ------------------------------ |
| Q, K, V       | Derived from input             |
| QK^T          | Similarity between tokens      |
| softmax       | Normalized attention weights   |
| Multiply by V | Aggregate context              |
| Multi-head    | Capture multiple relationships |

