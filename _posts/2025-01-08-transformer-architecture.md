---
title: "Understanding Transformer Architecture: The Brains Behind Modern AI"
date: 2025-01-08
categories:
  - Large Languague Model Architectures
tags:
  - Data Science
  - Generative AI
---

Transformers have revolutionized the world of AI, powering models like GPT, BERT, and T5. But what actually goes on inside these models?

In this blog post, we’ll explore the motivation behind transformers, the three main architectural variants (full transformer, encoder-only, and decoder-only), their best applications, and how each component works — with practical examples and diagrams.

---

## 🚀 Motivation

Before transformers, models like RNNs and LSTMs were used for sequential data. These models process tokens **one at a time**, which limits parallelism and struggles with long-range dependencies.

**Transformers** changed the game by using **self-attention**, which allows each token to directly consider **all other tokens in the sequence simultaneously** — capturing relationships between words, no matter how far apart they are.

This makes transformers faster to train, more scalable, and dramatically more powerful.

---

## 🧱 Transformer Architectures: Full, Encoder-Only, Decoder-Only

There are three main variants of the transformer architecture, each optimized for different types of tasks:

| Architecture     | Example Models            | Best For                                |
| ---------------- | ------------------------- | --------------------------------------- |
| Full Transformer | T5, BART, MarianMT        | Translation, summarization, multimodal  |
| Encoder-Only     | BERT, RoBERTa             | Classification, sentence similarity, QA |
| Decoder-Only     | GPT-2/3/4, LLaMA, ChatGPT | Text generation, chat, code completion  |

### ✅ Why Each Fits Its Application

- **Encoder-only** models create **contextual representations** of text — ideal for understanding tasks.
- **Decoder-only** models generate text **token by token**, making them ideal for chat, storytelling, coding, and completion.
- **Encoder–decoder** models separate input and output — perfect for **sequence-to-sequence** tasks like translation.

---

## 📊 Architecture Diagrams

### 📘 Full Transformer (Encoder–Decoder)

```text
Input Tokens (e.g. English)
   ↓
[Encoder Stack]
   ↓
Context Representations
   ↓
[Decoder Stack with Cross-Attention]
   ↓
Output Tokens (e.g. French)
```

### 📗 Encoder-Only Transformer

```text
Input Tokens
   ↓
[Encoder Stack]
   ↓
Contextual Embeddings → used for classification or sentence-level tasks
```

### 📙 Decoder-Only Transformer

```text
Prompt/Input Tokens
   ↓
[Decoder Stack with Masked Self-Attention]
   ↓
Autoregressive Output → one token at a time
```

---

## 🧱 Detailed Encoder Stack (Used in BERT, T5)

Each encoder layer in the stack follows this sequence:

### 1. Token Embeddings + Positional Encoding

- Maps each word/token into a vector of size `d_model`
- Adds positional encoding (sinusoidal or learned) to represent order

### 2. Multi-Head Self-Attention

- For each token, calculates attention over **all other tokens**
- Uses learned projections to compute $$Q$$ (query), $$K$$ (key), and $$V$$ (value)
- Computes:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- Attention from multiple heads are concatenated and passed through $$W^O$$

### 3. Add & LayerNorm (Residual Block 1)

- Adds the attention output to the original input and applies LayerNorm

### 4. Feedforward Neural Network (FFN)

- Two-layer MLP with GELU or ReLU
  $$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$
- Applied independently to each token

### 5. Add & LayerNorm (Residual Block 2)

- Adds FFN output to input of FFN and normalizes again

This block is repeated $$N$$ times to build deeper semantic understanding.

---

## 🧱 Detailed Decoder Stack (Used in GPT, T5, BART)

Each decoder layer includes all of the above **plus masking and cross-attention**:

### 1. Token Embeddings + Positional Encoding

- Same as encoder
- During training, uses **shifted input** (e.g. "Translate: The cat sleeps" → "Le", then "Le chat", etc.)

### 2. Masked Multi-Head Self-Attention

- Tokens can **only attend to earlier positions** (causal masking)
- Prevents the model from seeing future tokens

### 3. Add & LayerNorm (Residual Block 1)

- As in encoder

### 4. Cross-Attention (Encoder-Decoder only)

- Query from decoder attends to K/V from **encoder output**
- Allows decoder to access input sentence meaning (e.g., translation)

### 5. Add & LayerNorm (Residual Block 2)

- Same structure as above

### 6. Feedforward Neural Network (FFN)

- Same as encoder

### 7. Add & LayerNorm (Residual Block 3)

- Final residual normalization

Decoder layers are also repeated $$N$$ times for generation depth.

We'll now explore each component.

---

## 1️⃣ Token Embeddings + Positional Encoding

### **What it does:**
Converts each input token into a vector and adds a positional signal. 
- Each word (like “The”, “cat”, “sleeps”) is converted into a vector of numbers that captures the meaning of the word. 
- Adds information about word order (like “first”, “second”, etc.) to each word vector.

### **Why it's needed:**
- The model can't work directly with text. It needs a numerical understanding of words. These embeddings capture similarities, like "cat" and "dog" has a larger similarity than "cat" and "sleep". 
- Transformers don't know order by default. So "the cat sleeps" and "sleeps cat the" would look the same without this. Positional encoding tells the model who came before and after.

### **Example:**
For the sentence `"The cat sleeps"`:
- Token embeddings:
  ```python
  E_token = [e_The, e_cat, e_sleeps]  # shape: (3, d_model)
  ```
- Positional embeddings (learned or sinusoidal):
  ```python
  P = [p_0, p_1, p_2]
  ```
- Final input:
  ```python
  X = E_token + P
  ```

---

## 2️⃣ Multi-Head Self-Attention

### **What it does:**
Each word looks at **all the other words** in the sentence and decides **how much attention to pay** to each of them.

### **Why it's needed:**
This helps the model understand context. Multi-head means this is done in multiple ways at once. One head might look at subject-verb, another at adjectives, etc.

### **How it works:**
- Compute query (Q), key (K), value (V) vectors:
  $$ Q = XW^Q, \quad K = XW^K, \quad V = XW^V $$
- Compute attention weights:
  $$ \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) $$
- Multiply by V and combine heads:
  ```python
  Attention = softmax(Q @ K.T / sqrt(d_k)) @ V
  MultiHead = Concat(Head1, Head2, ...) @ W^O
  ```

### **Example:**
If the word is "sleeps", attention helps it realize that "cat" is the subject performing the action.

---

## 3️⃣ Add & LayerNorm (Post-Attention)

### **What it does:**
Adds the attention output back to the input and normalizes. 

### **Why it's needed:**
This helps the model not forget the original word info, and makes training stable and fast.

### **Formula:**

$$\text{LayerNorm}(X + \text{MultiHead}(X)) $$

---

## 4️⃣ Feedforward Neural Network (FFN)

### **What it does:**
It runs a mini (2-layer) Multi-Layer Perceptron (MLP) on each word’s vector separately to help the model better understand and refine the meaning of that word in context. MLP is a specific type of FNN made of fully connected layers with nonlinear activations (e.g., ReLU or GELU).

### **Why it's needed:**
It helps the model transform the meaning of each word based on what it has learned — like going from "noun" to "subject", or sharpening what was learned in attention.

### **Formula:**
$$ \text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2 $$

### **Example:**
After attention figures out that "cat" is important to "sleeps", the FFN updates the "sleeps" vector to reflect this.

---

## 5️⃣ Add & LayerNorm (Post-FFN)

### **What it does:**
Adds FFN output to its input and applies LayerNorm again.

### **Why it's needed:**
Same reason: keeps info flowing well and helps avoid forgetting or over-correcting.

---

## 🔁 Stack of Layers

This process is repeated N times. Each layer refines the understanding.
- Early layers learn local syntax.
- Mid layers capture sentence structure.
- Later layers understand abstract meaning.

---

## 🛠️ **Key Advantages of Transformer Architecture**

- **Parallel processing** of inputs.
- **Long-range dependency handling**.
- **Scalable** to massive parameter counts.
- Suitable for **pretraining + finetuning paradigm** in LLMs.

## ⚡ **Limitations**

- **Quadratic attention complexity** w.r.t. sequence length.
- **input sequence length limit** (e.g., 2048–128K tokens).
- Memory and compute heavy.

## ✅ Final Thoughts

Decoder-only transformers like GPT have proven incredibly powerful, as they can perform many tasks **just by clever prompting**, without needing a full encoder-decoder structure.

Still, encoder-only and full transformer models are valuable in **understanding tasks** and **structured input-output tasks**, respectively.

Understanding these architectures is essential to mastering LLMs like GPT, BERT, Claude, Gemini, LLaMA, and beyond.

For further inquiries or collaboration, feel free to contact me at [my email](mailto\:tungvutelecom@gmail.com).







