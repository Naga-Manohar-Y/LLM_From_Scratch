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
![Architecture](https://github.com/Naga-Manohar-Y/LLM_From_Scratch/blob/main/images/Transformer%20architecture.png)
- Consists of two submodules: **Encoder** and **Decoder**
- **Encoder** processes input text into numerical representations (embeddings)
- **Decoder** generates the output text from these embeddings


## **2. Working with Text Data**  
### 2.1 Understanding word embeddings  
### 2.2 Tokenizing text  
### 2.3 Converting tokens into token IDs  
### 2.4 Adding special context tokens  
### 2.5 BytePair encoding  
### 2.6 Data sampling with a sliding window  
### 2.7 Creating token embeddings  
### 2.8 Encoding word positions  

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
