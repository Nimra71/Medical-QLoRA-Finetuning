# -*- coding: utf-8 -*-

!pip install -q unsloth transformers datasets accelerate peft trl bitsandbytes

!pip install -q "fsspec==2025.12.0"

import torch
torch.cuda.is_available(), torch.cuda.get_device_name(0)

from datasets import load_dataset

full_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
split_dataset = full_dataset.train_test_split(test_size=100, seed=3407)

raw_train = split_dataset["train"]
eval_dataset = split_dataset["test"]

def format_example(example):
    question = example["question"]
    context = " ".join(example["context"]["contexts"])
    decision = example["final_decision"]
    long_answer = example["long_answer"]

    text = f"""### Instruction:
Answer the following medical question with YES, NO, or MAYBE, then explain briefly.

### Question:
{question}

### Context:
{context}

### Response:
Answer: {decision.upper()}
Explanation: {long_answer}"""
    return {"text": text}

dataset = raw_train.map(format_example)
dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

eval_dataset.save_to_disk("pubmedqa_eval_holdout")
print(f"Training on {len(dataset)} examples, holding out {len(eval_dataset)} for evaluation")

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=20,
    max_steps=600,   # was 200
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="medical-qlora-output",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=training_args,
)

trainer.train()

model.save_pretrained("medical_lora_adapter")
tokenizer.save_pretrained("medical_lora_adapter")

from google.colab import files
import shutil

shutil.make_archive("medical_lora_adapter", 'zip', "medical_lora_adapter")
files.download("medical_lora_adapter.zip")

FastLanguageModel.for_inference(model)

prompt = """### Instruction:
Answer the following medical question.

### Question:
What are common symptoms of asthma?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

from datasets import load_from_disk

eval_dataset = load_from_disk("pubmedqa_eval_holdout")

def build_prompt(example):
    question = example["question"]
    context = " ".join(example["context"]["contexts"])
    return f"""### Instruction:
Answer the following medical question with YES, NO, or MAYBE, then explain briefly.

### Question:
{question}

### Context:
{context}

### Response:
"""

def get_answer(m, tok, prompt):
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    outputs = m.generate(**inputs, max_new_tokens=100, temperature=0.3, top_p=0.9)
    text = tok.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def extract_label_v3(text):
    t = text.lower()
    if "answer: yes" in t: return "yes"
    if "answer: no" in t: return "no"
    if "answer: maybe" in t: return "maybe"
    return "unclear"

FastLanguageModel.for_inference(model)

ft_correct = 0
results = []
for ex in eval_dataset:
    prompt = build_prompt(ex)
    raw = get_answer(model, tokenizer, prompt)
    pred = extract_label_v3(raw)          # <-- fixed: was extract_label
    correct = (pred == ex["final_decision"])
    ft_correct += correct
    results.append({"question": ex["question"], "gold": ex["final_decision"], "ft_pred": pred, "raw": raw})

print(f"Fine-tuned model: {ft_correct}/{len(eval_dataset)} correct")

del model
torch.cuda.empty_cache()

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(base_model)

base_correct = 0
for i, ex in enumerate(eval_dataset):
    prompt = build_prompt(ex)
    raw = get_answer(base_model, tokenizer, prompt)
    pred = extract_label_v3(raw)
    correct = (pred == ex["final_decision"])
    base_correct += correct
    results[i]["base_pred"] = pred
    results[i]["base_raw"] = raw

print(f"Base model: {base_correct}/{len(eval_dataset)} correct")
print(f"Fine-tuned model: {ft_correct}/{len(eval_dataset)} correct")

for r in results[:5]:
    print("GOLD:", r["gold"])
    print("BASE RAW:", r["base_raw"][:150])
    print("-" * 50)
