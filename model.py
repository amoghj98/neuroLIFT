import torch
from torch import nn

class LLMForClassification(nn.Module):
	def __init__(self, base_model, n_labels=2):
		super(LLMForClassification, self).__init__()
		self.base_model = base_model  # The pre-trained Llama model
		for param in self.base_model.parameters():
			param.requires_grad = False
		self.hidden_size = self.base_model.config.hidden_size  # Hidden size of Llama
		print(self.hidden_size)
		print(n_labels)
		print(type(self.base_model.config))
		print(self.base_model.config.vocab_size)
		self.classifier = nn.Linear(self.hidden_size, n_labels)  # Classification head

	def forward(self, input_ids, attention_mask=None, labels=None):
		# Get outputs from the base model
		outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
		print(outputs.last_hidden_state.shape)
		# Extract the hidden states of the first token ([CLS]-like behavior)
		hidden_states = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
		# Pass the hidden states through the classification head
		logits = self.classifier(hidden_states)  # Shape: (batch_size, num_labels)
		# Compute loss if labels are provided
		loss = None
		if labels is not None:
			loss_fn = nn.CrossEntropyLoss()
			loss = loss_fn(logits, labels)
		return {"loss": loss, "logits": logits}