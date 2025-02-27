# **GPT Model from Scratch**  

This project is an implementation of a GPT model built entirely from scratch, inspired by *LLM from Scratch* by Sebastian. It covers key concepts like attention mechanisms, transformer architectures, and fine-tuning large language models for real-world applications.  

## **Table of Contents**  

| Topic | Description | README Section | Code File |
|--------|-------------|--------------|-----------|
| **Understanding Large Language Models** | Introduction to transformers, tokenization, and training objectives. | [Read More](#1-understanding-large-language-models) | [📜 Code](path/to/model_architecture.py) |
| **Working with Text Data** | Covers tokenization, word embeddings, special tokens, and data sampling. | [Read More](#2-working-with-text-data) | [📜 Code](path/to/text_preprocessing.py) |
| **Coding Attention Mechanisms** | Explains self-attention, causal masking, and multi-head attention. | [Read More](#3-coding-attention-mechanisms) | [📜 Code](path/to/attention.py) |
| **Implementing a GPT Model from Scratch** | Step-by-step implementation of a GPT model, including transformer blocks and text generation. | [Read More](#4-implementing-a-gpt-model-from-scratch-to-generate-text) | [📜 Code](path/to/gpt_model.py) |
| **Pretraining on Unlabeled Data** | Covers loss functions, decoding strategies, and loading pre-trained weights. | [Read More](#5-pretraining-on-unlabeled-data) | [📜 Code](path/to/pretraining.py) |
| **Finetuning for Text Classification** | Adapts the model for supervised tasks like spam detection, adding classification heads, and loss calculation. | [Read More](#6-finetuning-for-text-classification) | [📜 Code](path/to/finetune_classification.py) |
| **Instruction Finetuning** | Covers supervised instruction tuning, dataset preparation, and response extraction. | [Read More](#7-instruction-finetuning) | [📜 Code](path/to/rlhf_finetune.py) |

---

## **1. Understanding Large Language Models**  
### **What is an LLM?**
- A Large Language Model (LLM) is a deep neural network trained on vast amounts of text data to understand and generate human-like text. LLMs use the transformer architecture, which enables them to focus on different parts of the input using an attention mechanism. These models, trained via next-word prediction, power applications like chatbots, text summarization, and code generation.
- LLMs are widely used for:
  - Text generation
  - Machine translation
  - Sentiment analysis
  - Summarization
  - Question answering
  - Conversational AI (e.g., ChatGPT, Gemini, Claude)
### **Stages of Building and Using LLMs**
LLM training generally involves two key stages:

- **Pretraining** – Training on a massive dataset to learn general language structures using next-word prediction.
- **Finetuning** – Adapting the pretrained model to specific tasks using labeled datasets (e.g., instruction tuning or classification).

This two-step approach allows LLMs to be customized for specific applications while leveraging the knowledge learned from large-scale text corpora.

### **Transformer Architecture**
Most modern LLMs rely on the **transformer architecture**, introduced in the 2017 paper *Attention Is All You Need*. The original transformer was developed for machine translation.

**Architecture**

<img src="https://github.com/Naga-Manohar-Y/LLM_From_Scratch/blob/main/images/Transformer%20architecture.png" alt="Architecture" width="400" height="300">

- Consists of two submodules: **Encoder** and **Decoder**
- **Encoder** processes input text into numerical representations (embeddings)
- **Decoder** generates the output text from these embeddings

### **GPT Architecture**
- GPT architecture is relatively simple. It's just the decoder part without the encoder. Since decoder-style models like GPT generate text by predicting text one word at a time, they are considered a type of **autoregressive model**. 
- Autoregressive models incorporate their previous outputs as inputs for future predictions. Consequently, in GPT, each new word is chosen based on the sequence that precedes it, which improves coherence of the resulting text.
- GPT models, though designed for next-word prediction, unexpectedly perform translation—a phenomenon called "**emergent behavior.**" This arises from exposure to multilingual data, enabling diverse tasks without specialized training, showcasing the power of large-scale generative models.
### **Building a large language model in 3 stages**
<img src="https://camo.githubusercontent.com/a17472f25db0af2e7a72700cf3e994b48a61405931b54111ed4d62cbe0371216/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f6d656e74616c2d6d6f64656c2e6a7067" alt="Stages of building a LLM" width="400" height="200">

## **2. Working with Text Data**  
### **Word Embeddings**
- Deep neural networks can't process raw text directly, as it must be converted into numerical form. Embeddings map words or other discrete data into continuous vector space, enabling neural networks to handle text, images, or audio efficiently.

<img src="https://camo.githubusercontent.com/5aedb0a406bf9d298b251f57e1fbafb7bea9f993d16488675f2da6d419754505/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830325f636f6d707265737365642f30322e77656270" alt="Types of Embeddings" width="400" height="200">

<img src="https://camo.githubusercontent.com/1af3219e9329179a800d59e13ac04e32113cd4cd645ed3e805a48a3d49bf8996/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830325f636f6d707265737365642f30332e77656270" alt="Text Embeddings" width="200" height="200">

### **Preparing Embeddings for LLMs**

When training a Large Language Model (LLM), we need to convert raw text into a numerical format that the model can process. This involves several key steps:

**Tokenizing Text**   
Before converting words into numerical representations, we split text into tokens. 

<img src="https://camo.githubusercontent.com/241f7a302c33bc1e8156e7d0b153caae8728f2c9cd03884487c05d931fd88be2/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830325f636f6d707265737365642f30352e77656270" alt="Text Embeddings" width="300" height="100">

A tokenizer breaks down input text into:  
- Words ("Hello world" → ["Hello", "world"])
- Subwords ("unfamiliar" → ["unfam", "iliar"])
- Characters (if needed)
- Special tokens ([BOS], [EOS], [PAD], [UNK]):
  - [BOS] (Beginning of sequence)
  - [EOS] (End of sequence)
  - [PAD] (Padding to equalize sequence lengths)
  - [UNK] (Unknown words that don’t exist in the vocabulary)

Each token is then mapped to a unique integer (token ID) using a vocabulary.   

<img src="https://camo.githubusercontent.com/11a0a59ffbb8eb8e6a90eb4ea7706e4be0d7ed9b53cadd0d31f676af267866c0/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830325f636f6d707265737365642f30392e776562703f313233" alt="Text Embeddings" width="300" height="200">

**Byte Pair Encoding (BPE)** - GPT’s Tokenization Method
Why BPE? LLMs need to handle words outside their vocabulary (out-of-vocabulary words). Instead of storing every possible word, Byte Pair Encoding (BPE) breaks words into subwords.

<img src="https://camo.githubusercontent.com/5938dff392e5cb7404d2636e4d7157fceb4c36ecf57a2173001bd3edf22234da/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830325f636f6d707265737365642f31312e77656270" alt="Text Embeddings" width="300" height="200">

- This allows the model to generalize words it hasn't explicitly seen during training.
- GPT-2 uses OpenAI’s tiktoken library, which implements BPE in Rust for better efficiency.

**Preparing Input-Target Pairs for Training**.  
To train an LLM, we need to structure the data properly:

- Chunking text into smaller sequences.
- Next-word prediction: 
The model predicts the next word given the previous words.
Example:

- `Input:  ["The", "cat", "sat", "on"]`        
`Target: ["cat", "sat", "on", "the"]`

- The target is just a right-shifted version of the input.
Using DataLoaders in PyTorch:
- The Dataset and DataLoader classes load the data efficiently in mini-batches.

**Creating Token Embeddings (Converting Tokens into Vectors):**

- Since token IDs are just numbers, we need to convert them into meaningful numerical representations:
- Convert token IDs into 256-dimensional embedding vectors (GPT-3 uses 12,288 dimensions).

- Embedding layer: Maps token IDs to high-dimensional embedding vectors.
Example: If a token ID is 3, it retrieves the corresponding row from the embedding matrix.

<img src="https://camo.githubusercontent.com/30c75dce5178bdb6f53a37899c08b44c92eff2306c38beef19a831dc3770fc00/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830325f636f6d707265737365642f31362e776562703f313233" alt="Text Embeddings" width="300" height="200">

- Why embeddings? They allow words with similar meanings to have similar numerical representations.

**Encoding Word Positions (Positional Embeddings)**

- Embedding layer convert IDs into identical vector representations regardless of where they are located in the input sequence:
<img src="https://camo.githubusercontent.com/2659e7bc3eed30da2e6a0e6adc3143d6240c2759e5315481b21877eb12de47e1/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830325f636f6d707265737365642f31372e77656270" alt="Text Embeddings" width="300" height="200">

- LLMs process words without knowing their order, which can cause problems. To fix this, we add positional embeddings, which provide a sense of word order.

- There are two types of positional embeddings:
  - **Absolute Positional Embeddings** (used in GPT models):
    Assigns a fixed embedding to each position in a sequence.

- These embeddings are optimized during training.
  - **Relative Positional Embeddings**: Instead of storing absolute positions, it encodes distances between words.  
"cat" and "sat" may have a distance of 1.  
"cat" and "mat" may have a distance of 3

**Final Processing Before Training**
- To create the input embeddings used in an LLM, we simply add the token and the absolute positional embeddings:

<img src="https://camo.githubusercontent.com/730badacd85e476130cab5a98990d3c616b4333921096c576c31a50e7c0ca627/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830325f636f6d707265737365642f31392e77656270" alt="Text Embeddings" width="200" height="300">


## **3. Coding Attention Mechanisms**  
### 3.1 The problem with modeling long sequences  
### 3.2 Capturing data dependencies with attention mechanisms  
### 3.3 Attending to different parts of the input with self-attention  
- **3.3.1 A simple self-attention mechanism without trainable weights**  
- **3.3.2 Computing attention weights for all input tokens**  
### 3.4 Implementing self-attention with trainable weights  
- **3.4.1 Computing the attention weights step by step**  
- **3.4.2 Implementing a compact SelfAttention class**  
### 3.5 Hiding future words with causal attention  
- **3.5.1 Applying a causal attention mask**  
- **3.5.2 Masking additional attention weights with dropout**  
- **3.5.3 Implementing a compact causal self-attention class**  
### 3.6 Extending single-head attention to multi-head attention  
- **3.6.1 Stacking multiple single-head attention layers**  
- **3.6.2 Implementing multi-head attention with weight splits**  

## **4. Implementing a GPT Model from Scratch to Generate Text**  
### 4.1 Coding an LLM architecture  
### 4.2 Normalizing activations with layer normalization  
### 4.3 Implementing a feed forward network with GELU activations  
### 4.4 Adding shortcut connections  
### 4.5 Connecting attention and linear layers in a transformer block  
### 4.6 Coding the GPT model  
### 4.7 Generating text  

## **5. Pretraining on Unlabeled Data**  
### 5.1 Evaluating Generative Text Models  
- **5.1.1 Using GPT to Generate Text**  
- **5.1.2 Calculating the Text Generation Loss: Cross Entropy and Perplexity**  
- **5.1.3 Calculating the Training and Validation Set Losses**  
### 5.2 Training an LLM  
### 5.3 Decoding Strategies to Control Randomness  
- **5.3.1 Temperature Scaling**  
- **5.3.2 Top-k Sampling**  
- **5.3.3 Modifying the Text Generation Function with Above Strategies**  
### 5.4 Loading and Saving the Weights in PyTorch  
### 5.5 Loading the Pre-trained Weights from OpenAI  

## **6. Finetuning for Text Classification**  
### 6.1 Finetuning  
### 6.2 Data Preparation (Spam Data)  
### 6.3 Creating Data Loaders  
### 6.4 Initializing the Model with Pre-trained Weights  
### 6.5 Adding a Classification Head  
### 6.6 Calculating the Classification Loss and Accuracy  
### 6.7 Finetuning the Models on Supervised Data  
### 6.8 Using the LLM as a Spam Classifier  

## **7. Instruction Finetuning**  
### 7.1 Introduction to instruction finetuning  
### 7.2 Preparing Dataset for Supervised Instruction Finetuning  
### 7.3 Organizing data into training batches  
- **7.3.1 Creating Target Token IDs for Training**  
### 7.4 Creating data loaders for an instruction dataset  
### 7.5 Loading a pretrained LLM  
### 7.6 Finetuning the LLM on instruction data  
### 7.7 Extracting and saving responses  
### 7.8 Evaluating the finetuned LLM  

---
