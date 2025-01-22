# data.py

import nltk
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np

# Download NLTK data if not already present
nltk.download('all')

# Tokenizer function using NLTK
def tokenize(text, max_length=200):
    tokens = nltk.word_tokenize(text.lower())
    # Simple approach: keep first max_length tokens
    tokens = tokens[:max_length]
    return tokens

# Convert tokens to indices in a naive way (build a small vocab)
# We will do dynamic vocab building in code below for demonstration

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_length=200):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = tokenize(text, self.max_length)
        
        # Convert tokens to indices
        input_ids = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            input_ids += [self.word2idx["<PAD>"]] * (self.max_length - len(input_ids))
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)  # For binary sentiment
        return input_ids, label

def build_vocab(all_texts, vocab_size=20000):
    # Very naive frequency-based vocabulary building
    from collections import Counter
    token_counter = Counter()
    
    for text in all_texts:
        tokens = nltk.word_tokenize(text.lower())
        token_counter.update(tokens)
    
    # Most common 'vocab_size - 2' tokens (we'll reserve 2 for <PAD> and <UNK>)
    most_common = token_counter.most_common(vocab_size - 2)
    vocab = ["<PAD>", "<UNK>"] + [wc[0] for wc in most_common]
    
    word2idx = {w: i for i, w in enumerate(vocab)}
    return word2idx

def get_dataloaders(batch_size=32, max_length=200, vocab_size=20000):
    # 1. Load IMDB dataset
    dataset = load_dataset("imdb")
    
    # 2. We'll use train split for training, test split for validation
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # 3. Build vocabulary from the training set
    train_texts = train_data["text"]
    word2idx = build_vocab(train_texts, vocab_size=vocab_size)
    
    # 4. Create Dataset objects
    train_dataset = IMDBDataset(
        texts=train_data["text"], 
        labels=train_data["label"], 
        word2idx=word2idx,
        max_length=max_length
    )
    test_dataset = IMDBDataset(
        texts=test_data["text"], 
        labels=test_data["label"], 
        word2idx=word2idx,
        max_length=max_length
    )
    
    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, word2idx
