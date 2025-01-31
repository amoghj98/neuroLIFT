import torch
from torch import nn

class LLMForClassification(nn.Module):
	def __init__(self, base_model, n_labels=2, tokenizer=None, len_tokensizer=3072):
		super(LLMForClassification, self).__init__()
		self.l1_out = 512
		self.l2_out = 32
		self.tokenizer = tokenizer
		# self.loss_fn = nn.CrossEntropyLoss()
		self.loss_fn = nn.BCELoss()
		self.base_model = base_model  # The pre-trained Llama model
		for param in self.base_model.parameters():
			param.requires_grad = False
		self.hidden_size = len_tokensizer  # Hidden size of Llama
		self.classifier = nn.Sequential(
			nn.Linear(self.hidden_size, self.l1_out),  # Classification head
			nn.ReLU(),
			nn.Linear(self.l1_out, self.l2_out),
			nn.Sigmoid(),
			# nn.Linear(self.l2_out, n_labels),
			nn.Linear(self.l2_out, 1),
			nn.Sigmoid()
		)

	def forward(self, input_ids, attention_mask=None, labels=None):
		# Get outputs from the base model
		outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
		hidden_states = outputs.logits[:, -1, :]
		# hidden_states = torch.flatten(outputs.logits, start_dim=0, end_dim=1)
		# print(hidden_states.shape, flush=True)
		# print(outputs.last_hidden_state.shape)
		# Extract the hidden states of the first token ([CLS]-like behavior)
		# hidden_states = outputs['last_hidden_state'][:, 0, :]  # Shape: (batch_size, hidden_size)
		# hidden_states = nn.functional.normalize(outputs.logits.sum(dim=1), dim=1)  # Shape: (batch_size, hidden_size)
		# hidden_states = hidden_states / hidden_states.sum(dim=0)
		# print(outputs.logits.shape, flush=True)
		# print(self.base_model.config.hidden_size, flush=True)
		# Pass the hidden states through the classification head
		logits = self.classifier(hidden_states)  # Shape: (batch_size, num_labels)
		# print(logits.shape, flush=True)
		# Compute loss if labels are provided
		loss = None
		if labels is not None:
			# loss = self.loss_fn(logits, labels)
			loss = self.loss_fn(logits, labels[:, None].type(torch.float))
		# if self.tokenizer is not None:
		# 	print(self.tokenizer.decode(outputs.label_tokens, skip_special_tokens=True), flush=True)
		return {"loss": loss, "logits": logits}