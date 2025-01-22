# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMSentimentClassifier
from data import get_dataloaders

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Get DataLoaders
    train_loader, test_loader, word2idx = get_dataloaders(
        batch_size=64,
        max_length=200,
        vocab_size=20000
    )
    
    vocab_size = len(word2idx)
    model = LSTMSentimentClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=128)
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 2  # For demo; you can increase
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device).unsqueeze(1)  # shape [batch_size, 1]
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    # Simple evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids).squeeze()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save the model
    torch.save({
        "model_state_dict": model.state_dict(),
        "word2idx": word2idx
    }, "sentiment_model.pt")

if __name__ == "__main__":
    train_model()
