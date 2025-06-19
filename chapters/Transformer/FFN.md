# Feed Forward Neural Network (FFN) layer
After the self-attention mechanism gives a **context-rich representation** of each word, we further refine that representation using a Feed Forward Neural Network.

> A two-layer neural network applied to each word independently.

### 🎯 Purpose:
* Add **non-linearity** to the model.
* Allow **individual token transformation** — **independently** of others.
* Enhance expressive power of the model.

> Every token passes through the same FFN, but individually.

### ⚙ Structure: 2 Fully Connected (Dense) Layers
Formula:
$$
\large FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
$$

where:
* $x$: input vector (e.g., output from attention)
* $W_1, b_1$: weights and biases for the first layer
* $W_2, b_2$: weights and biases for the second layer
* ReLU: a non-linear activation function  

#### 🧠 Step-by-Step (Intuition)
Suppose:
Let’s say the output from self-attention for a word is:
$$\large x=[0.5, −1.2, 0.8]$$

#### 🔸 Step 1: First Linear Transformation
This step is like stretching or rotating the vector into a **higher-dimensional space**:

$$
\large h = xW_1 + b_1
$$

Let’s say we increase the dimension from 3 → 6:

$$
\large h=[0.1, −0.3, 2.1, 0.5, −1.1, 0.7]
$$

#### 🔸 Step 2: Apply ReLU
ReLU stands for Rectified Linear Unit:

$$
\large ReLU(z)=max(0,z)
$$

This removes **negative values** (introduces non-linearity):

$$
\large h'  =[0.1, 0, 2.1, 0.5, 0, 0.7]
$$

#### 🔸 Step 3: Second Linear Transformation
Now reduce it back to the original dimension (e.g., 6 → 3):

$$
\large Output=h'W_2 + b_2 = [1.5, 0.6, −0.1]
$$

#### 🧮 Summary of the FFN
| Step                        | Purpose                                          |
| --------------------------- | ------------------------------------------------ |
| Linear (W₁ + b₁)            | Expands dimension, mixes information             |
| ReLU                        | Introduces non-linearity                         |
| Linear (W₂ + b₂)            | Projects back to original dimension              |
| Same weights for all tokens | Efficient & consistent token-wise transformation |

#### 🧠 Layman Analogy
Imagine each word is a ball of clay (the output from attention).

The FFN:

* Stretches it (linear transform)

* Presses it to flatten negative noise (ReLU)

* Shapes it back into its final refined form

#### 📦 Output Shape
If input = `[batch_size, sequence_length, d_model]`
Then FFN keeps the same shape — just changes internal values per token.


#### 🧮 Example FFN
Let’s define:

Input vector (from LayerNorm): x = [0.1, -1.2, 0.4, 1.1]

Hidden size = 8 → Output size = 4


```python
# Step: Feed Forward Layer
import numpy as np
np.random.seed(42)
W1 = np.random.rand(4, 8)
b1 = np.random.rand(8)
W2 = np.random.rand(8, 4)
b2 = np.random.rand(4)

# Input vector
x = np.array([0.1, -1.2, 0.4, 1.1])

# First linear layer + ReLU
hidden = np.maximum(0, x @ W1 + b1)

# Second linear layer
ffn_output = hidden @ W2 + b2

print("FFN Output Vector:", ffn_output)
```

    FFN Output Vector: [1.88645838 3.62081468 3.3789379  4.04562467]
    

#### Diagram or flow


```python
Input Embedding → + Positional Encoding
           ↓
    Multi-Head Self-Attention
           ↓
      + Residual (Add)
           ↓
     → Layer Normalization
           ↓
    → Feed Forward Network (FFN)
           ↓
      + Residual (Add again)
           ↓
     → Layer Normalization (again)
           ↓
         Output of Encoder Layer

```


```python
import torch #  PyTorch base library
import torch.nn as nn # For neural network layers (like Linear)
import torch.nn.functional as F # For functions like ReLU

```

    Input vector: tensor([[-0.1400, -0.7194,  0.5050, -1.4887, -0.2266, -0.4161, -0.8963,  0.6397]])
    FFN output: tensor([[-0.3762, -0.1323,  0.1411, -0.6321, -0.1465,  0.0608,  0.0794, -0.5634]],
           grad_fn=<AddmmBackward0>)
    


```python
# Simulate output from attention layer for 1 token with d_model = 8
x = torch.randn(1, 8)  # Shape: (batch_size=1, d_model=8)

# This simulates a single token’s embedding vector after self-attention.
# It’s an 8-dimensional vector (just like d_model = 8 in Transformer).
# The shape is (1, 8) — meaning 1 token with 8 features.
```


```python
# Define the Feed Forward Network manually
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # First layer expands
        self.linear2 = nn.Linear(d_ff, d_model)  # Second layer compresses back

    def forward(self, x):
        x = self.linear1(x)     # Linear Transformation 1
        x = F.relu(x)           # Non-linearity
        x = self.linear2(x)     # Linear Transformation 2
        return x


# FFN is a class inheriting from nn.Module (standard in PyTorch).
# d_model = input dimension (e.g., 8).
# d_ff = hidden layer dimension (e.g., 32).
# linear1: transforms input from d_model → d_ff
# linear2: transforms back from d_ff → d_model


## Forward Pass: How the FFN processes input
### Linear Layer 1  
# h=xW1 + b1
        # Increases the vector size for higher-dimensional processing.

#### ReLU
        # ReLU(z)= max(0, z
#Applies non-linearity:

### Linear Layer 2
# Compresses the representation back to d_model size.
# Output = hW2  b2
```


```python
# Initialize FFN
d_model = 8     # input and output dimension
d_ff = 32       # intermediate (hidden) dimension
ffn = FFN(d_model, d_ff)

# FFN Initialization
# We specify the size of the input/output (d_model = 8)
# Intermediate layer (d_ff = 32) is larger to allow more transformation space.
```


```python
# Run input through FFN
output = ffn(x)

print("Input vector:", x)
print("FFN output:", output)

# Pass Input Through the FFN
# Takes the single 8D input vector x
# Applies the 2-layer transformation
# Outputs another 8D vector (same shape, refined content)
```

#### 📌 Explanation
* `d_model = 8`: like the embedding size.
* `d_ff = 32`: typically 2–4x larger than `d_model` in Transformer papers.
* FFN does:
1. Project up: `Linear(d_model → d_ff)`
2. Apply `ReLU`
3. Project down: `Linear(d_ff → d_model)`

> This FFN is applied **independently to each token**, but with **shared weights** across the sequence.

#### 🧠 What This Simulates
Each token in a sentence (like "India", "is", "great") **gets its own vector** from self-attention.
This FFN layer:

* Applies the **same neural network** to each of those vectors
* Adds **non-linearity and transformation capacity**
* Refines the token representation for better downstream use (e.g., classification or generation)


```python

```
