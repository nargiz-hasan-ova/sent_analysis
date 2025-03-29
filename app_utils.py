# app_utils.py

import csv
import urllib.request
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def preprocess_text(text):
    """
    Preprocess text to match how the CardiffNLP models were trained on tweets.
    Replaces @mentions with '@user' and URLs with 'http'.
    """
    new_text = []
    for t in text.split():
        if t.startswith('@') and len(t) > 1:
            t = '@user'
        elif t.startswith('http'):
            t = 'http'
        new_text.append(t)
    return " ".join(new_text)

def load_model_and_labels(task_name):
    """
    Load the 'cardiffnlp/twitter-roberta-base-{task_name}' model, tokenizer,
    and label mapping from the official TweetEval repository.
    """
    model_id = f"cardiffnlp/twitter-roberta-base-{task_name}"

    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Load label mapping (e.g., 0 -> negative, 1 -> neutral, 2 -> positive for sentiment)
    labels = []
    mapping_url = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task_name}/mapping.txt"
    with urllib.request.urlopen(mapping_url) as f:
        html = f.read().decode('utf-8').strip().split("\n")
        csvreader = csv.reader(html, delimiter='\t')
        for row in csvreader:
            if len(row) > 1:
                labels.append(row[1])

    return tokenizer, model, labels

def load_hate_model():
    return load_model_and_labels("hate")

def get_prediction(tokenizer, model, labels, text):
    """
    Given a tokenizer, model, and labels, run inference on text, returning
    a list of (label, probability) sorted by descending probability.
    """
    text = preprocess_text(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output.logits.detach().numpy()[0]
    scores = softmax(scores)

    ranking = np.argsort(scores)[::-1]
    result = [(labels[i], float(scores[i])) for i in ranking]
    # Example: [("positive", 0.93), ("neutral", 0.06), ("negative", 0.01)]
    return result
