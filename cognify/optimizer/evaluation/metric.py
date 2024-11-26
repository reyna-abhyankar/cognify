from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


def f1_score_set(pred, label):
    # Calculate true positives, false positives, and false negatives
    true_positives = len(label & pred)
    false_positives = len(pred - label)
    false_negatives = len(label - pred)

    if true_positives == 0:
        return 0

    # Calculate F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def f1_score_ordered(pred, label):
    matches = sum(1 for gt, pred in zip(label, pred) if gt == pred)
    precision = matches / len(pred)
    recall = matches / len(label)
    
    if precision + recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


import re
import string
import unicodedata


def normalize_text(s):
    # Normalize Unicode characters
    s = unicodedata.normalize("NFD", s)
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove articles (a, an, the)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Fix extra whitespaces
    s = " ".join(s.split())
    return s

def f1_score_str(pred, label):
    """F1 score for two strings

    Will first tokenize the strings by space and calculate F1 score
    """
    label = set(normalize_text(label).split())
    pred = set(normalize_text(pred).split())
    return f1_score_set(pred, label)
