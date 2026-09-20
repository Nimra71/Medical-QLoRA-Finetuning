# 🩺 Medical QLoRA Fine-Tuning with Unsloth

Fine-tuning **Llama 3 8B** on medical question-answering using **QLoRA** — trained on a free Google Colab GPU, evaluated on held-out clinical questions.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-Llama%203%208B-yellow)
![Method](https://img.shields.io/badge/Method-QLoRA-orange)
![Framework](https://img.shields.io/badge/Framework-Unsloth-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training](#training-the-model)
  - [Inference](#running-inference)
  - [Evaluation](#running-evaluation)
- [Evaluation Methodology](#-evaluation-methodology)
- [Folder Structure](#-folder-structure)
- [Performance Optimization](#-performance-optimization)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔍 Overview

This project fine-tunes **Llama 3 8B (4-bit quantized)** on the **PubMedQA** medical question-answering dataset using **QLoRA** (Quantized Low-Rank Adaptation), a parameter-efficient fine-tuning method. The model is trained to answer clinical research questions with a structured **YES / NO / MAYBE** judgment plus a supporting explanation — mirroring how PubMedQA itself is labeled.

The entire pipeline — training, adapter saving, and evaluation — runs on a **free Google Colab GPU** (T4), using only **0.5% of the model's total parameters** during fine-tuning.

---

## 📊 Results

The fine-tuned model was evaluated on **100 held-out PubMedQA questions** the model never saw during training. Each answer was checked against PubMedQA's ground-truth clinical decision label.

| Metric | Result |
|---|---|
| **Held-out accuracy** | **74 / 100 (74%)** |
| Held-out set size | 100 questions (never seen in training) |
| Base model | Llama 3 8B (4-bit) |
| Trainable parameters | ~0.5% of total model parameters |
| Training environment | Free Google Colab GPU (T4) |

> 📝 **Note on methodology:** Accuracy is measured by extracting the model's structured `Answer: YES/NO/MAYBE` output and comparing it against PubMedQA's `final_decision` gold label on a held-out split. A rigorous base-vs-fine-tuned baseline comparison (using few-shot prompting for a fair comparison) is planned — see [Future Improvements](#-future-improvements).

---

## ✨ Key Features

- 🏥 Medical question-answering model fine-tuned on real clinical research abstracts (PubMedQA)
- ⚡ 4-bit quantization for low memory usage
- 🎯 Parameter-Efficient Fine-Tuning (PEFT) using LoRA adapters
- 🚀 Accelerated training via the Unsloth framework
- 💻 Runs entirely on free Google Colab GPUs — no paid compute required
- 🔬 Trains only **0.5%** of total model parameters
- ✅ Includes a held-out evaluation pipeline with measured accuracy, not just qualitative examples

---

## 🛠 Tech Stack

| Component | Technology Used |
|---|---|
| **Base Model** | Llama 3 8B (4-bit) |
| **Fine-Tuning Method** | QLoRA |
| **Framework** | Unsloth |
| **Libraries** | Transformers, TRL, PEFT, BitsAndBytes |
| **Dataset** | [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (`pqa_labeled`) |
| **Environment** | Google Colab GPU (T4) |

---

## ⚙️ How It Works

1. Load Llama 3 8B in 4-bit quantized mode
2. Attach LoRA adapters to the model's attention and MLP layers
3. Split PubMedQA into training data and a **held-out evaluation set** (never trained on)
4. Fine-tune on the training split, teaching the model to output a structured `Answer: YES/NO/MAYBE` judgment followed by a clinical explanation
5. Save the trained LoRA adapter
6. Evaluate on the held-out set and measure accuracy against ground-truth labels

---

## 📦 Installation

```bash
pip install unsloth transformers datasets accelerate peft trl bitsandbytes
```

---

## 🚀 Usage

### Training the Model

Run the training notebook end-to-end. The key training call:

```python
trainer.train()
```

This fine-tunes the LoRA adapters on the PubMedQA training split using QLoRA.

### Running Inference

After training, test the model on a new medical question:

```python
prompt = """### Instruction:
Answer the following medical question with YES, NO, or MAYBE, then explain briefly.

### Question:
What are common symptoms of asthma?

### Response:
"""
```

The fine-tuned model generates a structured, domain-specific medical judgment and explanation.

### Running Evaluation

The notebook includes a held-out evaluation pipeline that:

1. Loads 100 questions never seen during training
2. Prompts the fine-tuned model for each one
3. Extracts the model's `YES / NO / MAYBE` judgment
4. Compares it against PubMedQA's ground-truth label
5. Reports overall accuracy

```python
print(f"Fine-tuned model: {ft_correct}/{len(eval_dataset)} correct")
```

---

## 🧪 Evaluation Methodology

To avoid data leakage, the dataset is split **before** any training occurs:

```python
full_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
split_dataset = full_dataset.train_test_split(test_size=100, seed=3407)

train_dataset = split_dataset["train"]   # used for fine-tuning
eval_dataset  = split_dataset["test"]    # held out — used only for evaluation
```

The model is trained to explicitly output its judgment (`Answer: YES/NO/MAYBE`) as the first line of its response, which is then programmatically extracted and scored against PubMedQA's labeled `final_decision` field.

---

## 📁 Folder Structure

```
Medical-QLoRA-Finetuning/
│── medical_qlora_project.ipynb   # Full training + evaluation notebook
│── README.md                     # Project documentation
```

---

## ⚡ Performance Optimization

| Technique | Purpose |
|---|---|
| 4-bit Quantization | Reduces memory usage significantly |
| LoRA Adapters | Trains only a small subset of parameters (~0.5%) |
| Gradient Accumulation | Simulates larger batch sizes on limited GPU memory |
| Unsloth Optimization | Speeds up training and lowers VRAM usage |

---

## 🔮 Future Improvements

- [ ] Add a fair **base-model vs. fine-tuned** comparison using few-shot prompting
- [ ] Train on larger medical datasets (e.g., MedQA, MedMCQA)
- [ ] Report precision/recall per class (YES / NO / MAYBE), not just overall accuracy
- [ ] Add a Gradio web interface for interactive testing
- [ ] Deploy the fine-tuned model as a medical Q&A chatbot
- [ ] Add safety filters and disclaimers for medical advice generation

---

## 📄 License

This project is licensed under the MIT License.
