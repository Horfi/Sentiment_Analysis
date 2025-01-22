# model.py

import torch
import torch.nn as nn

class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1):
        super(LSTMSentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        lstm_out, (h, c) = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_dim]
        # Use the final hidden state for classification
        hidden_out = h[-1]  # [batch_size, hidden_dim]
        logits = self.fc(hidden_out)  # [batch_size, 1]
        probs = self.sigmoid(logits)  # [batch_size, 1]
        return probs
