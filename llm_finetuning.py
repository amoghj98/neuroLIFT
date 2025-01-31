from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
import evaluate

import secret
from argConfig import config
from model import LLMForClassification
from dataset import neuroLIFTDataset

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
test_dataset = 'neuroLIFT_test.hdf5'
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
file = datasetPath + test_dataset
f = h5py.File(file, 'r')
prompts_test = [p.decode('utf-8', 'ignore') for p in f['prompts'][:]]
agent_position_test = f['agent_pos'][:]
agent_velocity_test = f['agent_vel'][:]
obst_position_test = f['obst_pos'][:]
obst_velocity_test = f['obst_vel'][:]
t_to_col_test = f['t_to_col'][:]
labels_test = f['labels'][:]
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
train_prompts = [f"{prompts[p_idxs[i]]} The robot's position is {agent_position[i]} and it's velocity is {agent_velocity[i]}, while the target's position is {obst_position[i]} and it's velocity is {obst_velocity[i]}. Answer only yes or no." for i in range(p_idxs.shape[0])]
#
p_idxs = np.random.randint(0, len(prompts_test), t_to_col_test.shape[0])
#
test_prompts = [f"{prompts_test[p_idxs[i]]} The robot's position is {agent_position_test[i]} and it's velocity is {agent_velocity_test[i]}, while the target's position is {obst_position_test[i]} and it's velocity is {obst_velocity_test[i]}. Answer only yes or no." for i in range(p_idxs.shape[0])]

# construct dataset for LLM
data = []
test = []
for i in range(len(train_prompts)):
    data.append({"input": train_prompts[i], "label":labels[i].item(), "label_token":label_to_token[labels[i].item()]})
for i in range(len(test_prompts)):
    test.append({"input": test_prompts[i], "label":labels_test[i].item(), "label_token":label_to_token[labels_test[i].item()]})

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
    # quantization_config=BitsAndBytesConfig(load_in_8bit=True),  # Use 8-bit precision (optional, saves memory)
)
#
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
#

# Tokenized dataset
tokenized_dataset = preprocess_fuction(data)
tokenized_testset = preprocess_fuction(test)

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
    eval_dataset=train_set,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
