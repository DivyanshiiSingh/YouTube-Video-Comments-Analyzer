from transformers import pipeline, AutoConfig, AutoTokenizer
from Youtube_comment_scraper import scrapeComments

def summarizeComments(API_Key, video_id):
  comments = scrapeComments(API_Key, video_id)
  total_comments = len(comments)
  print("Number of comments : ", total_comments)
  if len(comments) == 0 :
    print("No comments found!")
    return
  comments = " ".join(comments)

  summarizer = pipeline("summarization", model="facebook/bart-large-xsum")

  config = AutoConfig.from_pretrained("facebook/bart-large-xsum")
  max_len = config.max_position_embeddings

  try :
    summary = ''
    if (len(comments) < max_len) :
      summary = summarizer(comments, max_length=80, min_length=5, do_sample=False)[0]['summary_text']
    else :
      i = 0
      cnt = 1
      while i < len(comments) :
        print(f"summararizing chunk no. : {cnt}")
        text = comments[i : min(i + max_len, len(comments))]
        summary += " " + summarizer(text, max_length=80, min_length=5, do_sample=False)[0]['summary_text']
        i += max_len
        cnt += 1

      summary = summarizer(summary, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
    return summary
  except Exception as e :
    print(f"An error occurred : {e}")
