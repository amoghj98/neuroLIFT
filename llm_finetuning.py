from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
import evaluate

import secret
from argConfig import config
from model import LLMForClassification
from dataset import neuroLIFTDataset

import random
import copy

import time
import h5py
import sys
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader




def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_fuction(examples):
    inputs = [f"{example['input']}" for example in examples]
    labels = [1 if example['label'] else 0 for example in examples]
    label_tokens = [f"{example['label_token']}" for example in examples]
    return {"input_ids": tokenizer(inputs, truncation=True)["input_ids"], "labels": labels, "label_tokens" : label_tokens}

args = config()
#
datasetPath = '/scratch/gautschi/joshi157/finetuningDatasets/'
train_dataset = 'neuroLIFT.hdf5'
#
file = datasetPath + train_dataset
f = h5py.File(file, 'r')
prompts = [p.decode('utf-8', 'ignore') for p in f['prompts'][:]]
agent_position = f['agent_pos'][:]
agent_velocity = f['agent_vel'][:]
obst_position = f['obst_pos'][:]
obst_velocity = f['obst_vel'][:]
t_to_col = f['t_to_col'][:]
labels = f['labels'][:]
f.close()
#
label_to_token = {
    0: f"[LABEL_0]",
    1: f"[LABEL_1]",
}
token_to_label = {
    f"[LABEL_0]" : 0,
    f"[LABEL_1]" : 1,
}
##
p_idxs = np.random.randint(0, len(prompts), t_to_col.shape[0])
#
train_prompts = [f"A quadrotor drone is preparing to maneuver. The drone's max velocity is 17.53172 m/s, the drone's max acceleration is 1.25214 m/s^2. The drone's x-position must stay within [-1.2, 1.2] meters, the drone's y-position should stay with [0.4, 1.2] meters, and the drone's z-position must stay within [0.0, 1.4] meters. The drone observes the following: The obstacle is a moving ring, obstacle position is {obst_position[i]} meters, obstacle velocity is {obst_velocity[i]} m/s, drone position is {agent_position[i]}, drone velocity is {agent_velocity[i]}. Use kinematic feasibility is determine if {prompts[p_idxs[i]]} Your final answer should be a single token - either [LABEL_0] for No or [LABEL_1] for Yes followed by a period" for i in range(p_idxs.shape[0])]
#

# construct dataset for LLM
train_split = 0.9
data = []
for i in range(len(train_prompts)):
    data.append({"input": train_prompts[i], "label":labels[i].item(), "label_token":label_to_token[labels[i].item()]})

# split datasets
train_set_length = round(len(data) * train_split)
train_data = random.sample(data, train_set_length)
test_data = copy.deepcopy(data)
for d in train_data:
    test_data.remove(d)

# Download the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model, token=secret.huggingface_token)
#
model = AutoModelForSequenceClassification.from_pretrained(
    args.model,
    cache_dir=args.modelPath,
    token=secret.huggingface_token,
    device_map="auto",  # Automatically map to available GPUs
    id2label=label_to_token,
    label2id=token_to_label,
    num_labels=2,
    pad_token_id=2,
    ignore_mismatched_sizes=True,
    # quantization_config=BitsAndBytesConfig(load_in_8bit=True),  # Use 8-bit precision (optional, saves memory)
)
#
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
#
# for param in model.model.layers[:-4].parameters():
#     param.requires_grad = False

# Tokenized dataset
tokenized_dataset = preprocess_fuction(train_data)
tokenized_testset = preprocess_fuction(test_data)

#
train_set = neuroLIFTDataset(tokenized_dataset)
test_set = neuroLIFTDataset(tokenized_testset)

print(f'Train set length: {train_set.__len__()}')
print(f'Test set length: {test_set.__len__()}')

# Define training arguments and Trainer
training_args = TrainingArguments(
    output_dir="/scratch/gautschi/joshi157/results/",
    logging_dir="/scratch/gautschi/joshi157/logs/",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # save_steps=2000,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    learning_rate=3e-6,
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    # model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
