# Transformer
The **Transformer** is a deep learning architecture introduced in the 2017 paper “Attention Is All You Need” by Vaswani et al. It was developed for **natural language processing (NLP)** tasks but is now widely used in vision, audio, and multimodal AI.

> 🧠 Core idea: Instead of processing sequences step-by-step (like RNNs), the Transformer processes entire sequences in parallel using self-attention.

## Architecture Components
At its core, the **Transformer architecture** from “Attention Is All You Need” (Vaswani et al., 2017) consists of:

* **N stacked encoders**: For understanding input sequences.
* **N stacked decoders**: For generating output sequences.

These components work **together** in sequence-to-sequence (seq2seq) tasks like translation. But in many real-world models, we often use **only one of the two**, depending on the task.

### 🔹 Transformer Encoder
#### 📦 Function: Understand and represent the input
* **Input**: Tokenized sequence (e.g., "Delhi is the capital of India.")

* **Output**: Rich vector representations of the entire sentence

#### 🔧 Components in each encoder block:
1. **Multi-head self-attention**: Each word "attends" to all others
2. **Feed-forward network (FFN)**: Applies a non-linear transformation
3. **Residual + LayerNorm**: Stabilizes training

#### 🔄 Parallelizable
Since the encoder looks at all tokens at once, we can process entire sequences simultaneously (unlike RNNs/LSTMs).

### 🔹 Transformer Decoder
#### 📦 Function: Generate output one token at a time (auto-regressively)
* **Input**: Previously generated output tokens (e.g., during training: “Bonjour” → “Bonjour le”)

* **Output**: Next word prediction

#### 🔧 Components in each decoder block:
1. **Masked multi-head self-attention**: Only looks at previous tokens (prevents "cheating")
2. **Encoder-decoder attention**: Attends to the encoder’s output
3. **Feed-forward network**
4. **Residual + LayerNorm**

### 🔀 Key Difference in Behavior
| Feature                         | Encoder (BERT)                      | Decoder (GPT)                      |
| ------------------------------- | ----------------------------------- | ---------------------------------- |
| Attention                       | Unmasked (sees all tokens)          | Masked (sees only previous tokens) |
| Output                          | Embeddings (contextualized)         | Generated tokens                   |
| Use-case                        | Text understanding (classification) | Text generation                    |
| Encoder-Decoder Cross-Attention | ❌ Not used                          | ✅ Used in full seq2seq tasks       |




### 🧠 Why BERT Uses Only the Encoder
#### ✅ Task Type: Understanding-based
* BERT is trained using **masked language modeling**.
    * Example: “Delhi is the [MASK] of India.”
    * The model **predicts the missing word** based on **full context** (both left and right).

It needs **bidirectional context**, which is only possible with **unmasked attention** (from the encoder).

> 🔎 Think of BERT as a reader: it tries to understand full text before making a decision.

#### 🎯 Common BERT Tasks:
* Sentiment classification
* Question answering (e.g., SQuAD)
* Named entity recognition


### 🧠 Why GPT Uses Only the Decoder
#### ✅ Task Type: Generation-based
* GPT is trained using **causal language modeling** (predict next word).
    * Example: “Delhi is the capital of”
    * The model predicts: “India”
* Uses **masked attention** so it **can’t peek into future words**.
* The decoder structure suits **auto-regressive generation**: one word at a time.

> 🗣️ Think of GPT as a writer: it generates text, one token after another, based on what it has already written.

#### 🎯 Common GPT Tasks:
* Text completion
* Dialogue systems
* Story generation
* Code generation (e.g., Codex)

### 🔁 Encoder-Decoder in Full Transformers (e.g., T5, BART)
These models use **both** encoder and decoder:

* **Encoder**: Understands the input (e.g., a sentence in English)
* **Decoder**: **Generates** the output (e.g., sentence in French)

> Perfect for tasks like **translation, summarization, paraphrasing**

#### Summary Table
| Model   | Uses Encoder | Uses Decoder | Suitable For                   |
| ------- | ------------ | ------------ | ------------------------------ |
| BERT    | ✅            | ❌            | Understanding (classification) |
| GPT     | ❌            | ✅            | Generation (completion)        |
| BART/T5 | ✅            | ✅            | Sequence-to-sequence tasks     |


```python

```
