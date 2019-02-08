import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class GaitModel(nn.Module):

	def __init__(self, hidden_size, input_size, output_size, dropout_prob = 0.5):
		super(GaitModel, self).__init__()

		self.input_to_hidden = nn.Linear(input_size, hidden_size)
		self.hidden_to_logits = nn.Linear(hidden_size, output_size)
		self.dropout = nn.Dropout(p=dropout_prob)
		nn.init.xavier_uniform_(self.input_to_hidden.weight)
		nn.init.xavier_uniform_(self.hidden_to_logits.weight)

	def forward(self, x):
		hidden = self.input_to_hidden(x)
		relu = F.relu(hidden)
		logits = self.hidden_to_logits(self.dropout(relu))
		return logits
