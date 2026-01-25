# Medical QLoRA Fine-Tuning with Unsloth

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Gradio Dashboard](#gradio-dashboard)
- [Output & Analytics](#output--analytics)
- [Performance Optimization](#performance-optimization)
- [Folder Structure](#folder-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Introduction
This project demonstrates medical domain fine-tuning of a Large Language Model using QLoRA with the Unsloth framework.

---

## Key Features
- Medical question-answering fine-tuned model  
- 4-bit quantization for low memory usage  
- Parameter-Efficient Fine-Tuning (PEFT) using LoRA  
- Accelerated training with Unsloth  
- Runs on free Google Colab GPUs  
- Trains only 0.5% of total model parameters  

---

## Tech Stack

| Component | Technology Used |
|----------|----------------|
| Base Model | Llama 3 8B (4-bit) |
| Fine-Tuning Method | QLoRA |
| Framework | Unsloth |
| Libraries | Transformers, TRL, PEFT, BitsAndBytes |
| Dataset | PubMedQA |
| Environment | Google Colab GPU |

---

## How It Works
1. Load Llama 3 in 4-bit quantized mode  
2. Add LoRA adapters 
3. Fine-tune on PubMedQA medical Q&A  
4. Save trained adapter  

---

## Installation
```bash
pip install unsloth transformers datasets accelerate peft trl bitsandbytes

```


## Usage

After installing the dependencies, you can fine-tune the model using the provided notebook.

### Training the Model

Run the training cells in the notebook to begin fine-tuning:

```python
trainer.train()
```
This trains the LoRA adapters on the PubMedQA medical dataset using QLoRA.


Running Inference

After training, you can test the model with a medical question:

prompt = """### Instruction:
Answer the following medical question.

### Question:
What are common symptoms of asthma?

### Response:
"""


The fine-tuned model will generate a domain-specific medical response.

## Gradio Dashboard

A Gradio interface can be integrated to provide a simple web-based interface for interacting with the fine-tuned medical model.

This allows users to:

Enter medical questions

Receive AI-generated responses

Test model performance in real time

(This is an optional enhancement and not required for basic project completion.)

## Output & Analytics

During training, the following outputs can be observed:

Training loss decreasing over time, indicating learning

GPU memory usage optimized using 4-bit quantization

Improved quality of medical answers after fine-tuning

These metrics demonstrate successful domain adaptation of the model.

## Performance Optimization

Several optimization techniques were used to enable training on limited GPU memory:

## Technique	Purpose
4-bit Quantization	Reduces memory usage significantly
LoRA Adapters	Trains only a small subset of parameters
Gradient Accumulation	Simulates larger batch sizes
Unsloth Optimization	Speeds up training and lowers VRAM use

## Folder Structure

Medical-QLoRA-Finetuning/
│── medical_qlora_project.ipynb   # Training notebook
│── README.md                    # Project documentation


## Future Improvements

This project can be extended further by:

Training on larger medical datasets (e.g., MedQA, MedMCQA)

Adding automatic evaluation metrics

Deploying the model as a medical chatbot

Adding safety filters for medical advice generation

## License

This project is intended for educational and research purposes only.
The model should not be used for real-world medical diagnosis or treatment decisions.

