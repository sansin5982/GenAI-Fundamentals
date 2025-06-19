# Add & Norm (Residual + Layer Normalization)
After every major component (like **self-attention or FFN**), the Transformer applies:

> This step stabilizes training and improves performance.

#### ✅ Two steps:
1. **Add**: A **residual connection** (aka skip connection)
   * Add the original input vector (from embedding) to the self-attention output.
   * Helps preserve original information.
3. **Norm**: A layer normalization step
   * Normalize the vector across features (like BatchNorm, but per word).

The formula is:

$$
\large Output = LayerNorm(x) + Sublayer(x))
$$

#### 🧠 Layman Analogy
Imagine you're doing a group project (your main computation).

You don’t want to throw away your own ideas completely.

So you **add your original input** (x) back to the group’s suggestions (**Sublayer**(x)), then normalize to make things balanced.

This helps prevent:
* Forgetting important input features
* Too much influence from one component
### 🔢 Step-by-Step Breakdown
#### 🔸 Step 1: Residual Connection (Add)
$$
\large residual = x + Sublayer(x)
$$

* `x` is the input vector (e.g., from previous layer)
* `Sublayer(x)` is the output of **Self-Attention or FFN**
* This **“skip connection”** helps preserve the original information
* It improves **gradient flow** during training


#### 🧮 Example (simplified)
Let’s say for “India”, the:

Self-Attention output = `[0.5, 1.0, 0.8, 1.2]`

Input embedding (with position) = `[0.1, 1.3, 0.4, 1.5]`

#### Add (Residual)
residual_added = [0.5+0.1, 1.0+1.3, 0.8+0.4, 1.2+1.5]
                = [0.6, 2.3, 1.2, 2.7]

### 🔸 Step 2: Layer Normalization
We apply LayerNorm to the result:

$$
\large LayerNorm(z_i) = \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

where:

* $\mu$: mean of the vector
* $\sigma^2$: variance
* $\gamma, \beta$: learnable parameters (scale and shift)
* $\epsilon$: small value for numerical stability
> This ensures the output vector has **zero mean** and **unit variance**, which **stabilizes training**.


```python
import numpy as np

# Residual vector
x = np.array([0.6, 2.3, 1.2, 2.7])

# Layer Normalization
mean = x.mean()
std = x.std()
normalized = (x - mean) / (std + 1e-6)

print("Layer Normalized Vector:", normalized)

```

    Layer Normalized Vector: [-1.31007937  0.71458875 -0.59549062  1.19098125]
    

### Another example using torch


```python
import torch
import torch.nn as nn

# Input vector (e.g., token representation)
x = torch.randn(1, 8)  # batch size = 1, d_model = 8

print(x)
```

    tensor([[-1.4464, -1.0357, -0.4356, -1.9942, -0.5325, -0.4291, -0.4998, -0.3973]])
    


```python
# Let's simulate a sublayer output (e.g., from FFN or attention)
sublayer_output = torch.randn(1, 8)

print(sublayer_output)
```

    tensor([[ 1.2111,  2.4635,  1.0626, -0.7040, -1.1205,  0.1620,  1.2656,  0.4253]])
    


```python
# Step 1: Add residual
residual = x + sublayer_output
residual
```




    tensor([[-0.2354,  1.4278,  0.6270, -2.6982, -1.6530, -0.2670,  0.7659,  0.0280]])




```python
# Step 2: Apply LayerNorm
layer_norm = nn.LayerNorm(normalized_shape=8)
output = layer_norm(residual)

# Print
print("Original input x:", x)
print("Sublayer output:", sublayer_output)
print("After Add & Norm:", output)
```

    Original input x: tensor([[-1.4464, -1.0357, -0.4356, -1.9942, -0.5325, -0.4291, -0.4998, -0.3973]])
    Sublayer output: tensor([[ 1.2111,  2.4635,  1.0626, -0.7040, -1.1205,  0.1620,  1.2656,  0.4253]])
    After Add & Norm: tensor([[ 0.0121,  1.3344,  0.6977, -1.9460, -1.1150, -0.0131,  0.8082,  0.2215]],
           grad_fn=<NativeLayerNormBackward0>)
    

#### Why This Matters in Transformers
| Component      | Role                                             |
| -------------- | ------------------------------------------------ |
| Residual (Add) | Preserves input signal; improves gradient flow   |
| LayerNorm      | Stabilizes output distribution                   |
| Together       | Helps model **train deeper**, faster, and better |

#### 🔄 Where It’s Used
Every block in the Transformer has **two Add & Norm** layers:
* After Multi-Head Attention
* After Feed Forward Network (FFN)


```python

```
