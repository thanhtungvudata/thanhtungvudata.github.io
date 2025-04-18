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

In this blog post, we'll demystify the transformer architecture by walking through each component, how they relate to one another, and give practical examples to help you understand how it all fits together.

---

## 🚀 Motivation

Before transformers, models like RNNs and LSTMs were used for sequential data. These models process tokens **one at a time**, which limits parallelism and struggles with long-range dependencies.

**Transformers** changed the game by using **self-attention** to allow each token to look at the entire sequence in **parallel**, capturing relationships between words, no matter how far apart they are.

This makes transformers faster to train, more scalable, and dramatically more powerful.

---

## 📊 Transformer Diagram 

```text
Input Tokens
   ↓
Token Embeddings + Positional Encoding
   ↓
Multi-Head Attention
   ↓
Add & LayerNorm
   ↓
Feedforward Neural Network
   ↓
Add & LayerNorm
   ↓
(repeat N times)
   ↓
Final Output → (Used for prediction/classification/etc.)
```

N varies based on the models (e.g., 12 layers in GPT-2 small, 96 in GPT-4).

We'll now explore each component.

---

## 1️⃣ Token Embeddings + Positional Encoding

### **What it does:**
Converts each input token into a vector and adds a positional signal. Each word (like “The”, “cat”, “sleeps”) is converted into a vector of numbers that captures the meaning of the word. Adds information about word order (like “first”, “second”, etc.) to each word vector.

### **Why it's needed:**
- The model can't work directly with text — it needs a numerical understanding of words. These embeddings capture similarities, like “cat” and “dog” being closer than “cat” and “sleep”. 
- Transformers don’t know order by default — so “the cat sleeps” and “sleeps cat the” would look the same without this. Positional encoding tells the model who came before and after.

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
Each token looks at **all other tokens** and decides **who to pay attention to**.

### **Why it's needed:**
Captures context from anywhere in the sequence — essential for meaning.

### **How it works:**
- Compute query (Q), key (K), value (V) vectors:
  \[ Q = XW^Q, \quad K = XW^K, \quad V = XW^V \]
- Compute attention weights:
  \[ \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \]
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
Helps with gradient flow and keeps representations stable.

### **Formula:**
\[ \text{LayerNorm}(X + \text{MultiHead}(X)) \]

---

## 4️⃣ Feedforward Neural Network (FFN)

### **What it does:**
Applies a mini-MLP to each token independently.

### **Why it's needed:**
Transforms and refines token-level features.

### **Formula:**
\[ \text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2 \]

### **Example:**
After attention figures out that "cat" is important to "sleeps", the FFN updates the "sleeps" vector to reflect this.

---

## 5️⃣ Add & LayerNorm (Post-FFN)

### **What it does:**
Adds FFN output to its input and applies LayerNorm again.

### **Formula:**
\[ \text{LayerNorm}(x + \text{FFN}(x)) \]

---

## 🔁 Stack of Layers

This process is repeated multiple times. Each layer refines the understanding.

- Early layers learn local syntax.
- Mid layers capture sentence structure.
- Later layers understand abstract meaning.

---

## ✅ Final Thoughts

Transformers are deep but modular. Once you understand how token embeddings, attention, and FFNs work together, the entire architecture becomes intuitive.

Mastering this opens the door to understanding GPT, BERT, T5, and almost every state-of-the-art NLP model today.

For further inquiries or collaboration, please contact me at [my email](mailto:tungvutelecom@gmail.com).




