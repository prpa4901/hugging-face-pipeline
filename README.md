# hugging-face-pipeline

# Hugging Face Labs Overview

This README provides a comprehensive guide to setting up and using Hugging Face models with libraries like `transformers`, `langchain`, and `langchain_huggingface`. It covers how to work with pre-trained models, integrate them into LangChain workflows, and run inference efficiently on GPUs or CPUs.

---

## Table of Contents
1. [Introduction to Hugging Face](#introduction-to-hugging-face)
2. [Setting Up Hugging Face Models](#setting-up-hugging-face-models)
3. [Using Hugging Face Pipelines](#using-hugging-face-pipelines)
4. [Integrating Hugging Face with LangChain](#integrating-hugging-face-with-langchain)
5. [Running Models on GPU](#running-models-on-gpu)
6. [Files Downloaded During Model Loading](#files-downloaded-during-model-loading)
7. [Chaining Prompts and Models](#chaining-prompts-and-models)
8. [Code Examples](#code-examples)

---

## Introduction to Hugging Face

Hugging Face provides pre-trained models and tools for natural language processing (NLP) and machine learning (ML). Using the `transformers` library, you can:
- Perform tasks like text generation, summarization, and question answering.
- Access a wide variety of pre-trained models.
- Leverage tools like pipelines for rapid prototyping.

The integration with LangChain enables building workflows for tasks like dynamic prompting, chaining operations, and combining multiple ML models.

---

## Setting Up Hugging Face Models

1. **Install the required libraries**:
   ```bash
   pip install transformers langchain langchain_huggingface
   ```

2. **Load Pre-trained Models**:
   Use Hugging Face's `transformers` library to load models and tokenizers.

   Example:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_name = "gpt2"
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   ```

3. **Create a Pipeline**:
   ```python
   from transformers import pipeline

   text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
   ```

---

## Using Hugging Face Pipelines

Hugging Face pipelines provide a high-level API to use models for specific tasks. Supported tasks include:
- **Text Generation**
- **Summarization**
- **Question Answering**
- **Named Entity Recognition (NER)**

### Example: Text Generation
```python
from transformers import pipeline

text_gen_pipeline = pipeline("text-generation", model="gpt2")
response = text_gen_pipeline("Write a poem about the sea.", max_length=50)
print(response[0]['generated_text'])
```

---

## Integrating Hugging Face with LangChain

LangChain enables structured workflows by combining models, prompts, and chaining operations. You can use Hugging Face models directly in LangChain workflows via `HuggingFacePipeline`.

### Example: Wrap Hugging Face Pipeline in LangChain
```python
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a pipeline
text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
```

---

## Running Models on GPU

To leverage GPUs for faster inference:
1. **Specify the GPU Device**:
   ```python
   device = 0  # Use the first available GPU
   text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
   ```

2. **Alternative: Use `device_map`**:
   ```python
   text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
   ```

---

## Files Downloaded During Model Loading
When loading a model from Hugging Face, the following files are downloaded:

1. **`config.json`**: Model configuration (e.g., number of layers, attention heads).
2. **`model.safetensors`**: Model weights in the `safetensors` format.
3. **`generation_config.json`**: Configuration for text generation (e.g., `max_length`, `temperature`).
4. **`tokenizer_config.json`**: Configuration for the tokenizer.
5. **`vocab.json` and `merges.txt`**: Vocabulary and merge rules for tokenization.
6. **`tokenizer.json`**: Serialized tokenizer information.

These files are stored in the local Hugging Face cache and reused for subsequent runs.

---

## Chaining Prompts and Models
LangChain simplifies chaining prompts and models to create workflows.

### Example: Prompt Chaining
```python
from langchain_core.prompts import PromptTemplate

# Define a template
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

# Combine with Hugging Face pipeline
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    device=0,  # Use GPU
    pipeline_kwargs={"max_new_tokens": 100},
)

# Chain prompt and model
chain = prompt | gpu_llm

# Run the chain
response = chain.invoke({"question": "What is the capital of France?"})
print(response)
```

---

## Code Examples

### Simple Text Generation
```python
from transformers import pipeline

text_gen_pipeline = pipeline("text-generation", model="gpt2")
response = text_gen_pipeline("Once upon a time", max_length=50)
print(response[0]['generated_text'])
```

### LangChain Integration
```python
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Create a pipeline
gpt2_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=50)
llm = HuggingFacePipeline(pipeline=gpt2_pipeline)

# Use LangChain prompt template
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("What is the meaning of {concept}?")
chain = prompt | llm

# Run the chain
response = chain.invoke({"concept": "life"})
print(response)
```

---

## Summary
This guide demonstrates how to:
1. Set up Hugging Face models with the `transformers` library.
2. Use Hugging Face pipelines for text generation and other tasks.
3. Integrate Hugging Face models into LangChain workflows.
4. Run models on GPUs for faster inference.

With these tools, you can build and customize workflows for NLP and machine learning tasks efficiently. Let me know if you'd like further clarification or enhancements!

