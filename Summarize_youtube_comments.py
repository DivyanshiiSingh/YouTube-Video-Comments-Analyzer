from typing import final
from transformers import pipeline
from Youtube_comment_scraper import scrapeComments

def summarizeComments(API_Key, video_id):
  comments = scrapeComments('API_Key', 'video_id')
  concatenated_comments = " ".join(comments)
  
  summarizer = pipeline("summarization")
  
  # truncating the input text to a smaller size that is sure to be within the model's processing limit
  max_input_length = summarizer.model.config.max_position_embeddings - 2
  print(f'max_input_length : {max_input_length}, type : {type(max_input_length)}')
  
  try:
    to_summarize = concatenated_comments
    last_summary = ""
    curr_summary = ""
    final_summary = ""
    i = 1
  
    while len(to_summarize) > 0:
        to_summarize = to_summarize + last_summary
  
        if len(to_summarize) <= max_input_length:
            final_summary = summarizer(to_summarize, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
            break
  
        curr_summary = summarizer(to_summarize[0: max_input_length], max_length=100, min_length=5, do_sample=False)[0]['summary_text']
  
        to_summarize = to_summarize[max_input_length:]
        last_summary = curr_summary
        print(f'summary cycle : {i}')
        i += 1
    print(f'final summary : {final_summary}')
  
  except Exception as e:
    print(f"An error occurred: {e}")
