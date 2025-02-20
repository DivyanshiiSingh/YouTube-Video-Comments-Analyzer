from googleapiclient.discovery import build

def scrapeComments(API_Key, video_id):
  try:
    youtube = build('youtube', 'v3', developerKey=API_Key)
  except Exception as e:
    print(f"Error building API : {e}")
    raise

  comments = []
  try:
    results = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText',).execute()
  except Exception as e:
    print(f"Error getting comments : {e}")
    results = None

  while results:
      for item in results['items']:
          comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
          comments.append(comment)

      if 'nextPageToken' in results:
          results = youtube.commentThreads().list(
              part='snippet', videoId=video_id, textFormat='plainText', pageToken=results['nextPageToken']).execute()
      else:
          break
  return comments
