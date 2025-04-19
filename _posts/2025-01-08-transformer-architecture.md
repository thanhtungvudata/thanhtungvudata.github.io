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

#### **What it does:**
Converts each input token into a vector and adds a positional signal. 
- Each word (like “The”, “cat”, “sleeps”) is converted into a vector of numbers that captures the meaning of the word. 
- Adds information about word order (like “first”, “second”, etc.) to each word vector.

#### **Why it's needed:**
- The model can't work directly with text. It needs a numerical understanding of words. These embeddings capture similarities, like "cat" and "dog" has a larger similarity than "cat" and "sleep". 
- Transformers don't know order by default. So "the cat sleeps" and "sleeps cat the" would look the same without this. Positional encoding tells the model who came before and after.

#### **How it works:**

Each input token (like "The", "cat", "sleeps") is mapped to a vector using a learned **token embedding matrix**. These vectors represent the meaning of each token in a high-dimensional space. 
Let’s break it down: 

##### Token Embeddings:
Given:
- Vocabulary size:  $$V$$
- Embedding dimension: $$d$$
- Embedding matrix: $$E_{\text{token}} \in \mathbb{R}^{V \times d}$$

If your sentence is `["The", "cat", "sleeps"]`, and these map to token IDs `[12, 45, 230]`, then:

$$
X = \begin{bmatrix}
E_{\text{token}}[12] \\
E_{\text{token}}[45] \\
E_{\text{token}}[230]
\end{bmatrix} \in \mathbb{R}^{3 \times d}
$$

These vectors are **learned during training** via backpropagation.

##### Positional Encoding:
We add a vector to each token embedding that tells the model its position in the sequence. There are two variants:

- **Sinusoidal PE (fixed):**

$$
\text{PE}[p, 2i] = \sin\left(\frac{p}{10000^{2i/d}}\right), \quad
\text{PE}[p, 2i+1] = \cos\left(\frac{p}{10000^{2i/d}}\right)
$$

- **Learned PE:**
$$ E_{\text{pos}} \in \mathbb{R}^{L \times d} $$, where $$ L $$ is max sequence length.

##### Final Input:
For position $$p $$ with its implied token ID $$ t_p $$:

$$
X_p = E_{\text{token}}[t_p] + \text{PE}[p]
$$

This combined vector $$ X_p $$ which encodes both **what the word is** and **where it is** becomes the input to the first transformer layer.

### 2. Multi-Head Self-Attention

#### **What it does:**
Each word looks at **all the other words** in the sentence and decides **how much attention to pay** to each of them.

#### **Why it's needed:**
This helps the model understand context. Multi-head means this is done in multiple ways at once. One head might look at subject-verb, another at adjectives, etc. For example, if the word is "sleeps", attention helps it realize that "cat" is the subject performing the action.

#### **How it works:**

##### 1. Linear projections for Q, K, V
Each input token vector $$ x \in \mathbb{R}^{1 \times d} $$ of input token matrix $$X \in \mathbb{R}^{n \times d} $$ is transformed into:
- Query vector: $$ Q = xW^Q \in \mathbb{R}^{1 \times d_k} $$
- Key vector: $$ K = xW^K \in \mathbb{R}^{1 \times d_k} $$
- Value vector: $$ V = xW^V \in \mathbb{R}^{1 \times d_k} $$

where $$ W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k} $$ are learned weight matrices, and $$ d_k $$ is typically $$ d / h $$, with $$ h $$ being the number of heads.

##### 2. Compute attention scores
For each query-key pair, compute a score:

$$
\text{score}_{ij} = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}
$$

This measures how much token $$ i $$ should attend to token $$ j $$.

##### 3. Apply softmax
Convert scores to attention weights:

$$
\alpha_{ij} = \text{softmax}_j\left(\text{score}_{ij}\right)
$$

Each $$ \alpha_{ij} \in [0,1] $$, and $$ \sum_j \alpha_{ij} = 1 $$

##### 4. Weighted sum of values
Use the attention weights to combine the values:

$$
\text{output}_i = \sum_j \alpha_{ij} V_j
$$

This is the context-aware representation for token $$ i $$.

##### 5. Do this for multiple heads
Repeat the steps above $$ h $$ times, each with its own set of learned $$ W^Q, W^K, W^V $$ matrices.

##### 6. Concatenate and project
Concatenate the output of all heads and apply a final linear projection:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

where $$ W^O \in \mathbb{R}^{(h \cdot d_k) \times d} $$ is also learned. 

$$ W^O $$ determines how to combine the different "perspectives" from all attention heads into a single, unified vector that can be used by the next layer. It decides how much weight to give to each head's output, essentially blending them into a coherent representation for each token.

##### Intuition:
- **Queries ask questions**: “Who’s relevant to me?”
- **Keys answer**: “Here’s what I have to offer.”
- **Values carry content**.
- Multi-head = many ways to look at the same sentence — one head might focus on syntax, another on long-distance relationships.

Each output vector is a blend of others — how much it blends depends on the attention scores. That’s how the model learns context.

### 3. Add & LayerNorm (Residual Block 1)

#### **What it does:**
Adds the attention output back to the original input (residual connection), then applies layer normalization.

#### **Why it's needed:**
- Helps prevent vanishing gradients and training instability
- Preserves the original token information and combines it with what attention learned
- Keeps the scale of values consistent across layers

#### **How it works:**

$$
Z = \text{LayerNorm}(X + \text{MultiHead}(X))
$$

which **normalizes** the result using **layer normalization**, which adjusts the mean and variance of the combined vector.

### 4. Feedforward Neural Network (FFN)

#### **What it does:**
Each token’s vector (after attention) is passed through a small neural network — the same one for every position — to refine its representation. 

Each token goes through this same transformation separately — it's position-wise, not sequence-wide.

#### **Why it's needed:**
- Adds nonlinearity to the model
- Lets the model reprocess each token’s context-aware vector independently
- Helps the model abstract and compress information

For example, if the word "sleeps" attends to "cat" in attention, the FFN helps turn that into a refined idea like "subject performs action.

#### **How it works:**
It consists of two fully connected layers with an activation in between:

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

- $$ W_1 \in \mathbb{R}^{d \times d_{ff}} $$, $$ W_2 \in \mathbb{R}^{d_{ff} \times d} $$
- $$ d_{ff} $$ is usually 4x the size of $$ d $$ (embedding dimension)
- Activation is typically **ReLU** or **GELU** (more commonly used in transformers)

#### Intuition:
- Think of this as a mini brain that processes each token in isolation.
- While attention lets tokens talk to each other, FFN helps each token **reflect and reshape** what it just learned.


### 5. Add & LayerNorm (Residual Block 2)

- Adds FFN output to input of FFN and normalizes again

This block is repeated $$N$$ times to build deeper semantic understanding.

---

## 🧱 Detailed Decoder Stack (Used in GPT, T5, BART)

Each decoder layer includes all of the above **plus masking and cross-attention**:

### 1. Token Embeddings + Positional Encoding (in Decoder)

#### **What it does & Why it's needed:**
Same as the encoder: tokens are converted into vectors and combined with positional encodings to retain order.

#### **How it works (differences from encoder):**
- Token embeddings and positional encodings are constructed the **same way** as in the encoder
- The **key difference** is that the decoder uses **shifted inputs**:

```text
Target sequence:         ["The", "cat", "sleeps"]
Shifted decoder input:   ["<BOS>", "The", "cat"]
```

`<BOS>` stands for Beginning of Sequence. It’s a special token inserted at the start of the decoder input to indicate the start of generation.
- It has its own embedding like any other token.
- It helps the model know when and how to begin generating output.
- During inference, generation typically begins from <BOS> alone.

#### Why the shift?
The decoder generates tokens **one by one**, using only the **tokens that came before**. During training, we shift the input so that:
- At each position $$ p $$, the model sees tokens $$ < t_0, t_1, ..., t_{p-1} > $$
- And it is trained to predict $$ t_p $$

This teaches the model to learn **autoregressive generation** — i.e., predict the next word based only on previously generated ones.

### 2. Masked Multi-Head Self-Attention (in Decoder)

#### **What it does & Why it's needed:**
In the decoder, masked self-attention ensures that each token can only attend to **earlier tokens** — not to future ones.

- This is essential for **autoregressive generation**, where the model generates text one token at a time.
- Without masking, the model could "cheat" by peeking at the token it’s supposed to predict.
- Masking enforces strict **causality** in generation.

#### **How it works (differences from encoder):**
- In the encoder, attention is **fully visible** — each token can attend to all others.
- In the decoder, we apply a **causal mask** that blocks attention to future positions.

##### Steps:
1. Compute Q, K, V projections just like in the encoder:

   $$
   Q = XW^Q, \quad K = XW^K, \quad V = XW^V
   $$

2. Compute raw attention scores:

   $$
   \text{scores} = \frac{QK^T}{\sqrt{d_k}}
   $$

3. Apply **causal mask**:
   Set all positions $$ (i,j) $$ where $$ j > i $$ to $$-\infty$$:

   $$
   \text{scores}_{ij} = -\infty \text{ if } j > i
   $$

4. Apply softmax:

   $$
   \alpha_{ij} = \text{softmax}_j(\text{scores}_{ij})
   $$

5. Compute output:

   $$
   \text{output}_i = \sum_j \alpha_{ij} V_j
   $$

Each token only "looks left" — at the tokens that came before.

### 3. Add & LayerNorm (Residual Block 1)

- As in encoder

### 4. Cross-Attention (Encoder-Decoder only)

#### **What it does**
Cross-attention allows the decoder to **look at the encoder’s output** — i.e., the representation of the input sequence.
- Each token in the decoder can attend to **all positions of the input**.
- This provides context from the input sequence that guides the decoder's output.

#### **Why it's needed:**
- Without cross-attention, the decoder would only generate based on its own past outputs.
- In tasks like **translation**, the decoder needs to condition its next token prediction on the **entire input sentence**.
- Cross-attention lets the model align output tokens with relevant input tokens.

#### **How it works:**
- Similar to regular attention, but instead of $$Q$$, $$K$$, $$V$$ all coming from the decoder, only **$$Q$$** comes from the decoder, and **$$K$$** and **$$V$$** come from the encoder.

##### Steps:
1. Use the decoder’s hidden states to compute queries:

   $$
   Q = X_{\text{decoder}} W^Q
   $$

2. Use the encoder’s output (fixed after encoding) to compute keys and values:

   $$
   K = X_{\text{encoder}} W^K, \quad V = X_{\text{encoder}} W^V
   $$

3. Compute attention scores:

   $$
   \text{scores} = \frac{QK^T}{\sqrt{d_k}}
   $$

4. Apply softmax to get weights:

   $$
   \alpha_{ij} = \text{softmax}_j(\text{scores}_{ij})
   $$

5. Compute weighted sum of values:

   $$
   \text{output}_i = \sum_j \alpha_{ij} V_j
   $$

This output is then passed forward in the decoder layer.

#### Intuition:
- Think of encoder-decoder attention as a **bridge**.
- The decoder is generating text and asks: “What did the input say that’s relevant right now?”
- Cross-attention provides that answer, using the encoder’s understanding of the input.

This mechanism is what allows **sequence-to-sequence** models to perform tasks like summarization, translation, and more.



### 5. Add & LayerNorm (Residual Block 2)

- Same structure as above

### 6. Feedforward Neural Network (FFN)

- Same as encoder

### 7. Add & LayerNorm (Residual Block 3)

- Final residual normalization

Decoder layers are also repeated $$N$$ times for generation depth.

---

## 🔁 Stack of Layers

This process is repeated $$N$$ times. Each layer refines the understanding.
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
- **Input sequence length limit** (e.g., 2048–128K tokens).
- Memory and compute heavy.

## Why Models Like GPT (Decoder-Only) Can Do Translation, Summarization, Multimodal

### 🔍 How is GPT able to perform tasks like translation or summarization?

Although GPT is a **decoder-only transformer**, it can handle tasks traditionally associated with **encoder–decoder models** because of how it’s trained and how prompting works:

#### 1. Instruction Tuning
GPT models are trained on datasets that include examples of translation, summarization, Q&A, etc. These are framed as **text-in → text-out** tasks.
- Example: "Translate English to French: The cat sleeps →"
- The prompt encodes both the **task type** and **input context**.

#### 2. Unified Text Format
In decoder-only transformers:
- **Inputs and outputs are handled in the same sequence stream.**
- The model learns to complete tasks based on pattern recognition from massive pretraining + finetuning.
- The autoregressive setup means GPT is great at generating structured completions.

#### 3. Prompt Engineering
You can turn nearly any problem into a **single text string**, which GPT learns to respond to appropriately:
```text
Summarize: Climate change is accelerating due to... → Summary:
Translate: Hello, how are you? → French:
Describe this image: <image tokens> → A dog jumping over a fence.
```
This lets GPT solve problems **without separate encoder/decoder modules**.

#### 4. Multimodal Support via Tokenization
- In GPT-4 and other multimodal variants, **non-text data (like images)** are tokenized and embedded into the same input stream.
- This allows models to **reason over images + text together**, using the same decoder-only architecture.

## Why Encoder-Only and Full Transformer Models Are Still Valuable

### Encoder-Only Models (like BERT)
- Produce **rich contextual embeddings** for each token.
- Best used when **no generation is needed**, e.g.:
  - Classification (e.g. sentiment analysis)
  - Semantic similarity
  - Named entity recognition
- They’re bidirectional — can attend to **both left and right context**.

### Encoder–Decoder Models (like T5, BART)
- Best for **structured input–output mappings** like:
  - Translation
  - Summarization
  - Question answering
- Clear separation of input encoding and output decoding allows better alignment in sequence-to-sequence problems.
- The encoder provides **full comprehension of input**, while the decoder **generates target sequences** with cross-attention.

---

## ✅ Final Thoughts

Decoder-only transformers like GPT have proven incredibly powerful, as they can perform many tasks **just by clever prompting**, without needing a full encoder-decoder structure.

Still, encoder-only and full transformer models are valuable in **understanding tasks** and **structured input-output tasks**, respectively.

The choice depends on task structure and deployment goals.

Understanding these architectures is essential to mastering LLMs like GPT, BERT, Claude, Gemini, LLaMA, and beyond.

For further inquiries or collaboration, feel free to contact me at [my email](mailto\:tungvutelecom@gmail.com).








