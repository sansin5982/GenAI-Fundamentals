# 🔷 What is Cross-Attention?
### 📘 Definition:
**Cross-attention** allows the model to **attend to** a **different sequence** than its own.

> 🔄 In a Transformer decoder, **cross-attention lets the output (decoder) tokens focus on the encoder’s outputs** — instead of just themselves.

### 🧠 Layman Analogy: Teacher + Notes
Imagine you’re writing an answer (**decoder**) by referring to your class notes (encoder).
Each word you write in your answer depends on **specific parts of the notes** — that’s **cross-attention**.

| Type            | "Who looks at whom?"                                    |
| --------------- | ------------------------------------------------------- |
| Self-Attention  | Token looks at other tokens in same sequence            |
| Cross-Attention | Decoder token looks at encoder outputs (input sequence) |

## 📐 Math Behind Cross-Attention
Same attention formula:

$$
\large Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

But with a twist:

* Queries $\large Q$: comes from the **decoder**
* Keys and Values $K, V$: come from the **encoder outputs**

### 🔁 Where Is It Used?
| Model Part             | Uses Cross-Attention?                  |
| ---------------------- | -------------------------------------- |
| Encoder                | ❌ No                                   |
| Decoder (first block)  | ❌ Only self-attention                  |
| Decoder (second block) | ✅ Yes — **cross-attention** to encoder |



```python
import torch
import torch.nn.functional as F

# Encoder output: 3 tokens, d_model = 4
encoder_output = torch.randn(1, 3, 4)  # (batch_size, seq_len_enc, d_model)

# Decoder input (query): 2 tokens, d_model = 4
decoder_input = torch.randn(1, 2, 4)   # (batch_size, seq_len_dec, d_model)

# Linear layers to project Q, K, V
W_q = torch.nn.Linear(4, 4)
W_k = torch.nn.Linear(4, 4)
W_v = torch.nn.Linear(4, 4)

# Step 1: Generate Q from decoder input
Q = W_q(decoder_input)  # Shape: [1, 2, 4]

# Step 2: Generate K, V from encoder output
K = W_k(encoder_output)  # Shape: [1, 3, 4]
V = W_v(encoder_output)  # Shape: [1, 3, 4]

# Step 3: Compute attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / (4 ** 0.5)  # shape: [1, 2, 3]

# Step 4: Softmax to get attention weights
weights = F.softmax(scores, dim=-1)  # shape: [1, 2, 3]

# Step 5: Multiply with V to get final context vectors
context = torch.matmul(weights, V)   # shape: [1, 2, 4]

print("Attention weights:\n", weights)
print("Context vectors (cross-attention output):\n", context)
```

    Attention weights:
     tensor([[[0.4226, 0.2801, 0.2973],
             [0.2732, 0.4197, 0.3071]]], grad_fn=<SoftmaxBackward0>)
    Context vectors (cross-attention output):
     tensor([[[-0.3062, -0.8716,  0.6253,  0.0605],
             [-0.1689, -1.0340,  0.6341,  0.1202]]], grad_fn=<UnsafeViewBackward0>)
    

#### 🧠 What Just Happened?
* Each **decoder token** used a Query vector.
* It looked at **all encoder tokens** using Keys & Values.
* The result: a **contextual vector** that blends relevant encoder information for each decoder token.

#### 🎯 When is Cross-Attention Crucial?
| Task             | Why Cross-Attention Matters               |
| ---------------- | ----------------------------------------- |
| Translation      | Output (target) depends on input (source) |
| Summarization    | Decoder “reads” from input document       |
| Image captioning | Caption words attend to image features    |

#### 📌 Summary
| Concept          | Self-Attention           | Cross-Attention                      |
| ---------------- | ------------------------ | ------------------------------------ |
| Queries from     | Same sequence            | Decoder input                        |
| Keys/Values from | Same sequence            | Encoder output                       |
| Purpose          | Understand local context | Learn which input parts are relevant |

### Difference between Self-attention and Cross-attention
| Scenario        | Self-Attention                                         | Cross-Attention                                                |
| --------------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| 🧑‍🎓 Student   | Thinks by referring to **own thoughts and past words** | Thinks by **consulting a textbook**                            |
| 🗣️ Translation | Understanding English by analyzing the sentence itself | Generating French translation by referring to English sentence |
| 📝 Summary      | Captures sentence meaning by relating its own words    | Picks key phrases from an external document                    |

### 🧮 Mathematical Comparison
| Component        | Self-Attention              | Cross-Attention                  |
| ---------------- | --------------------------- | -------------------------------- |
| Query (Q)        | Comes from **input itself** | Comes from **decoder input**     |
| Key, Value (K,V) | Also from **input itself**  | From **encoder output**          |
| Formula          | $\text{Attention}(Q, K, V)$ | Same formula, but $Q \neq K = V$ |

### 🔢 Example Use-Case
#### ✳️ Self-Attention
> In a sentence like “India is great,” each word looks at other words:

* "India" attends to "is" and "great" to understand full meaning.

✳️ Cross-Attention
> In translation:

* While decoding “L’Inde est grande,” the decoder attends to the encoder's “India is great” to decide which word to generate next.

#### Self attention


```python
Q = W_q(x)
K = W_k(x)
V = W_v(x)
# x comes from same source (input or decoder)
```

#### Self attention



```python
Q = W_q(x)
K = W_k(x)
V = W_v(x)
# x comes from same source (input or decoder)
```

#### Tabular Comparison
| Feature          | Self-Attention                    | Cross-Attention                       |
| ---------------- | --------------------------------- | ------------------------------------- |
| Used in          | Encoder & Decoder (first block)   | Decoder (second block only)           |
| Query from       | Same sequence                     | Decoder input                         |
| Key & Value from | Same sequence                     | Encoder output                        |
| Goal             | Learn context in same sequence    | Learn mapping from source → target    |
| Computation      | $Q, K, V$ from same input         | $Q$ ≠ $K, V$ (from different sources) |
| Example          | Understanding sentence internally | Translation, summarization            |


#### 🧠 Summary in 1 Line
> **Self-attention** = "Understand myself"

> **Cross-attention** = "Refer to another source for help"


```python

```
