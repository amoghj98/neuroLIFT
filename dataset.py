import torch
from torch.utils.data import Dataset


class neuroLIFTDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.input_prompt = tokenized_dataset["input_ids"]
        self.labels = tokenized_dataset["labels"]
        self.label_tokens = tokenized_dataset["label_tokens"]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids" : torch.tensor(self.input_prompt[idx]),
            "labels" : torch.tensor(self.labels[idx]),
        }