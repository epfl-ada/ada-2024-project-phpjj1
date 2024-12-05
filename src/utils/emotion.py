# Lexicon source is (C) 2016 National Research Council Canada (NRC) and this package is for research purposes only. Source: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm As per the terms of use of the NRC Emotion Lexicon, if you use the lexicon or any derivative from it, cite this paper: Crowdsourcing a Word-Emotion Association Lexicon, Saif Mohammad and Peter Turney, Computational Intelligence, 29 (3), 436-465, 2013.

import nltk
import torch
from nrclex import NRCLex
from transformers import pipeline, AutoTokenizer

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

def sliding_window(text: str, window_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Creates overlapping windows from the input text
    
    Args:
        text (str): The input text to be split into windows
        window_size (int): The maximum number of tokens per window
        overlap (int): The number of overlapping tokens between consecutive windows
        
    Returns:
        list[str]: A list of text windows
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = ' '.join(words[start:start + window_size])
        chunks.append(chunk)
        start += window_size - overlap

        if start >= len(words):
            break

    return chunks



roberta_classifier = pipeline(
        "text-classification", 
        model="SamLowe/roberta-base-go_emotions", 
        top_k=28, # always return all 28 emotions
        tokenizer=AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions"),
        truncation=True,
        device=device
    )

distilbert_classifier = pipeline(
                    "text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base", 
                    top_k = None,
                    tokenizer=AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base"),
                    truncation=True,
                    device=device
                    )

# Required for nrclex
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
nltk.download('punkt', quiet=True)

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
    """
    results = roberta_classifier(text)
    results = results[0]

    return {result['label']: result['score'] for result in results}
    

def get_emotions_distilbert(text: str) -> dict[str, float]:
    """
    Get Ekman's 6 emotions and a neutral class from text using HuggingFace's DistilRoBERTa model.
    Returns emotions with their scores in a dict format.
    
    Example output:
    ```
    {'label': 'anger', 'score': 0.004419783595949411},
    {'label': 'disgust', 'score': 0.0016119900392368436},
    {'label': 'fear', 'score': 0.0004138521908316761},
    {'label': 'joy', 'score': 0.9771687984466553},
    {'label': 'neutral', 'score': 0.005764586851000786},
    {'label': 'sadness', 'score': 0.002092392183840275},
    {'label': 'surprise', 'score': 0.008528684265911579}
    ```
    """
    results = distilbert_classifier(text)
    results = results[0]

    return {result['label']: result['score'] for result in results}
    

def get_emotions_sliding_window(text: str, classifier_func, window_size: int = 500, overlap: int = 100) -> dict[str, float]:
    """
    Processes a long text using a sliding window approach and aggregates the emotion scores.
    
    Args:
        text (str): The input text to process.
        classifier_func (function): The emotion classifier function to use (e.g., `get_emotions_roberta`).
        window_size (int): Maximum tokens per window.
        overlap (int): Overlapping tokens between windows.
    
    Returns:
        dict[str, float]: Aggregated emotion scores.
    """
    # Split text into overlapping windows
    windows = sliding_window(text, window_size, overlap)
    aggregated_emotions = {}
    total_windows = len(windows)

    for window in windows:
        # Get emotions for each window
        emotions = classifier_func(window)
        for emotion, score in emotions.items():
            # Accumulate the scores
            aggregated_emotions[emotion] = aggregated_emotions.get(emotion, 0) + score

    # Normalize scores by the number of windows
    aggregated_emotions = {emotion: score / total_windows for emotion, score in aggregated_emotions.items()}
    return aggregated_emotions



if __name__ == "__main__":
    print(get_emotions_roberta("I'm so happy and excited about this!"))
    print(get_emotions_nrclex("I'm so happy and excited about this!"))