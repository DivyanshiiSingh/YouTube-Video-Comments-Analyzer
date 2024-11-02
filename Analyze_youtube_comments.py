# deploys model to analyze youtube comments

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
from googleapiclient.discovery import build
from Youtube_comment_scraper import scrapeComments

def analyzeComments(API_Key, video_id, model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    
    comments = scrapeComments(API_Key, video_id)
    
    print("Number of comments: ", len(comments))
    
    inputs = tokenizer(comments, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    
    for i, comment in enumerate(comments):
        print(f"Comment: {comment}")
        print(f"Sentiment: {sentiment_labels[predictions[i].item()]}")
    
    
    positive_comments_count = sum(predictions == 2).item()
    negative_comments_count = sum(predictions == 0).item()
    neutral_comments_count = sum(predictions == 1).item()
    
    total_comments = len(comments)
    positive_comments_percentage = (positive_comments_count / total_comments) * 100
    negative_comments_percentage = (negative_comments_count / total_comments) * 100
    neutral_comments_percentage = (neutral_comments_count / total_comments) * 100
    
    print(f'Total comments: {total_comments}')
    print(f'Positive comments: {positive_comments_percentage : .2f}%')
    print(f'Negative comments: {negative_comments_percentage : .2f}%')
    print(f'Neutral comments: {neutral_comments_percentage : .2f}%')
