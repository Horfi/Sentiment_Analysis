# inference.py

import torch
import nltk
from model import LSTMSentimentClassifier

nltk.download("punkt")  # ensure nltk sentence tokenizer is available

def load_model(model_path="sentiment_model.pt", device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    word2idx = checkpoint["word2idx"]
    
    model = LSTMSentimentClassifier(vocab_size=len(word2idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, word2idx

def predict_sentiment(text, model, word2idx, device="cpu", max_length=200):
    # Simple tokenize
    tokens = nltk.word_tokenize(text.lower())
    tokens = tokens[:max_length]
    input_ids = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    
    # Pad if needed
    if len(input_ids) < max_length:
        input_ids += [word2idx["<PAD>"]] * (max_length - len(input_ids))
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        score = model(input_tensor).item()  # single float
    return score

def color_for_score(score):
    """
    Given a sentiment score in [0.0, 1.0], return a color.
    We'll break it down as:
      [0.0, 0.2) -> red
      [0.2, 0.4) -> orange
      [0.4, 0.6) -> yellow
      [0.6, 0.8) -> gray
      [0.8, 1.0] -> green
    """
    if score < 0.2:
        return "red"
    elif score < 0.4:
        return "orange"
    elif score < 0.6:
        return "yellow"
    elif score < 0.8:
        return "gray"
    else:
        return "green"

def analyze_text(text, model, word2idx, device="cpu"):
    """
    Split input text into sentences, predict each sentence's sentiment, 
    color-code it, and compute overall average sentiment.
    Returns: 
      List of tuples (sentence, color, score)
      overall_score
    """
    sentences = nltk.sent_tokenize(text)
    sentence_results = []
    sum_score = 0.0
    
    for sent in sentences:
        score = predict_sentiment(sent, model, word2idx, device=device)
        clr = color_for_score(score)
        sentence_results.append((sent, clr, score))
        sum_score += score
    
    if len(sentences) > 0:
        overall_score = sum_score / len(sentences)
    else:
        overall_score = 0.0
    
    return sentence_results, overall_score
