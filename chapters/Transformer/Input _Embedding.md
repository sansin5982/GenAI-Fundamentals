## Step 1. Input Embedding (What is it?)
### 🧠 What does "embedding" mean?
In simple terms:
* Words are not numbers.
* But neural networks only understand **numbers**.
* So we convert words into **dense numeric vectors** called **embeddings**.

#### For example
* **"India is great"**

#### Tokenization
We tokenize it into words:

["India", "is", "great"]

Each of these is assigned a unique ID using a tokenizer vocabulary:

"India" → 1204  

"is"    → 305  

"great" → 761

So, the input becomes:

[1204, 305, 761]

#### Token Embedding
Each word gets mapped to a **vector of real numbers** using an embedding matrix. Here d_model = 4. 
In the context of Transformers (and most NLP models), d_model refers to the dimensionality of the embedding vectors — in simpler terms:

> 👉 d_model = 4 means every word (token) is represented as a 4-dimensional vector.

In real Transformer models like BERT or GPT, d_model is often **256**, **512**, or even **1024+** — meaning:

* Each word is turned into a **512-dimensional** or **1024-dimensional vector**.
* These high-dimensional embeddings allow the model to capture rich semantic patterns.

#### 🧮 2. Why powers of 2 like 256, 512?
Because:

* Memory alignment on modern GPUs is faster with powers of 2.
* Efficient matrix multiplications in deep learning frameworks (PyTorch, TensorFlow) work better with powers of 2.
* Empirically tested and used in research papers (e.g., Vaswani et al., 2017 used d_model = 512 in the original Transformer).

📊 Real-world use in models
| Model                    | Typical `d_model` |
| ------------------------ | ----------------- |
| BERT Base                | 768               |
| BERT Large               | 1024              |
| GPT-2 small              | 768               |
| GPT-3                    | 2048 – 12288      |
| Custom/Small Transformer | 128 – 512         |

#### 🧪 Intuition: Analogy
Imagine each word is a **container of meaning**.

* With `d_model = 4`, you can store **just basic attributes** (e.g., subject or verb).
* With `d_model = 512`, you can store **richer features** like:
    * tense
    * tone
    * syntax
    * semantic relationships
    * context history

#### 🧠 Tradeoff: Accuracy vs Resources
| `d_model` Size         | Pros                        | Cons                           |
| ---------------------- | --------------------------- | ------------------------------ |
| Small (e.g., 64)       | Fast, low memory            | Less accurate, loses context   |
| Medium (e.g., 256–512) | Balanced speed and power    | Used in many production models |
| Large (e.g., 1024+)    | High accuracy, deep context | Needs powerful GPU/TPU         |


| Token | Embedding (Vector)    |
| ----- | --------------------- |
| India | \[0.1, 0.3, 0.4, 0.5] |
| is    | \[0.2, 0.1, 0.6, 0.3] |
| great | \[0.4, 0.3, 0.9, 0.1] |


These come from a **learned embedding layer**.

> 📌 Mathematically:
If the **vocabulary size = V**, and **embedding dimension = d**, then:


The embedding matrix: 

$E \in \mathbb{R}^{V \times d}$

A word with index $i$ gets the embedding $E[i]$


E = [
  [0.1, 0.3, 0.4, 0.5],  
  [0.2, 0.1, 0.6, 0.3],  
  [0.4, 0.3, 0.9, 0.1]   
]
* This captures what the words mean (semantics), but not where they are in the sentence.
    * Firsr list is India
    * Second is is
    * Third is great


```python

```
