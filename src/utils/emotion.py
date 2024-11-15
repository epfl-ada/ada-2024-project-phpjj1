from transformers import pipeline, AutoTokenizer
from collections import Counter
import torch
import math



def model_and_tokenizer(model_name = "j-hartmann/emotion-english-distilroberta-base"):
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline('text-classification', model=model_name, top_k =None, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return classifier, tokenizer


def split_equal_chunks(text, tokenizer, max_size): 
    tokens = tokenizer.tokenize(text)
    tokens_len = len(tokens)
    num_chunks = (tokens_len + max_size -1)//max_size
    chunk_size = int(tokens_len/num_chunks)
    chunks = []
    for i in range(0, tokens_len, chunk_size):
        chunks.append(" ".join(tokens[i:i+chunk_size]))
    return chunks

def get_emotion(text,classifier, tokenizer):
    max_size = tokenizer.model_max_length - 10 
    chunks = split_equal_chunks(text, tokenizer,max_size)
    agg = Counter()
    for chunk in chunks:
        result = classifier(chunk)
        for r in result[0]:
            agg[r['label']] += r['score']
    
    tot = sum(agg.values())
    text_result = [ { 'label':label, 'score':agg[label]/tot } for label in agg]
    sort_result = sorted(text_result, key = lambda x: x['score'], reverse = True)
    return sort_result