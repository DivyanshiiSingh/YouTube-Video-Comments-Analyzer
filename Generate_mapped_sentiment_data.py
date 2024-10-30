# Using, GoEmotions dataset to train the mBERT model, so, need to map various emotions of GoEmotions
# to the three required emotions i.e. positive, negative and neutral

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from collections import Counter

ds = load_dataset("google-research-datasets/go_emotions", "simplified")

label_mapping = {
    'positive': {0, 1, 4, 15, 17, 18, 20},      # admiration, amusement, approval, gratitude, joy, love, optimism
    'negative': {2, 3, 9, 10, 11, 14, 25},      # anger, annoyance, disappointment, disapproval, disgust, fear, sadness
    'neutral': {6, 7, 22}                       # confusion, curiosity, realization
}

# label_mapping to map each integer ID to the corresponding category
id_to_category = {}
for category, ids in label_mapping.items():
    for id in ids:
        id_to_category[id] = category

def map_labels(label_ids):
    for label_id in label_ids:
        if label_id in id_to_category:
            return id_to_category[label_id]
    return 'neutral'  # Default to neutral if no emotions matched

# applying the mapping to create simplified labels
ds = ds['train']
comments = ds['text']
simplified_labels = [map_labels(label) for label in ds['labels']]

df = pd.DataFrame({'comment': comments, 'sentiment': simplified_labels})

def create_labelled_dataframe(sample_size) :
  if (sample_size > len(df)) :
    print(f'given sample size is larger than the whole dataset!, please give a size <= {len(df)}')
    return pd.DataFrame()
  return df.sample(sample_size)


dfDebugSample = df.sample(100)  # a small sample for quick checking of label distribution
label_counts = Counter(dfDebugSample['sentiment'])
print("Label distribution:", label_counts)
