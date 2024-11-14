# I think the emotion analysis is very important to have the emotions as attribute for each movie, describe the classes that exist and an initial analysis for research question 1 would be great so what are the most common emotional tones and how do they differ between genres

# Lexicon source is (C) 2016 National Research Council Canada (NRC) and this package is for research purposes only. Source: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm As per the terms of use of the NRC Emotion Lexicon, if you use the lexicon or any derivative from it, cite this paper: Crowdsourcing a Word-Emotion Association Lexicon, Saif Mohammad and Peter Turney, Computational Intelligence, 29 (3), 436-465, 2013.

import os
import nltk
from nrclex import NRCLex
from transformers import pipeline, AutoTokenizer
import torch

roberta_classifier = pipeline(
        "text-classification", 
        model="SamLowe/roberta-base-go_emotions", 
        top_k=28 # always return all 28 emotions
    )

# Initialize tokenizer to handle plot summaries that are too long
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

# Required for nrclex
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_emotions_nrclex(text: str) -> dict[str, float]:
    """
    Return a dictionary that looks like this:
    ```
    {'fear': 0.0, 'anger': 0.0, 'anticipation': 0.0, 'trust': 0.0, 'surprise': 0.0, 'positive': 0.5, 'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.5}
    ```
    """
    emotion_scores = NRCLex(text)
    return emotion_scores.affect_frequencies


def get_emotions_roberta(text: str) -> dict[str, float]:
    """
    Get all 28 emotions from text using HuggingFace's RoBERTa model, trained on Reddit data.
    Returns emotions with their scores in a dict format. 500MB model size.
    
    Example output:
    ```
    {'fear': 0.0, 'anger': 0.0, 'anticipation': 0.0, 'trust': 0.0, 'surprise': 0.0, 
     'positive': 0.5, 'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.5}
    ```
    """
    # Check if given text is really a string
    if not isinstance(text, str):
        return None

    # Use truncation for plot summaries that exceed the 514 token limit
    encoded_input = tokenizer(text, truncation=True, max_length=514, return_tensors="pt")
    results = roberta_classifier(encoded_input['input_ids'])
    results = results[0]

    return {result['label']: result['score'] for result in results}
    


    

if __name__ == "__main__":
    print(get_emotions_roberta("I'm so happy and excited about this!"))
    print(get_emotions_nrclex("I'm so happy and excited about this!"))



