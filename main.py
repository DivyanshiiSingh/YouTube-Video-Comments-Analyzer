from Fine_tune_mbert import fineTune
from Analyze_youtube_comments import analyzeComments
from Summarize_youtube_comments import summarizeComments

while True:
  choice = input("Hello! What are you intersted in ?\n1. Fine tune mBERT\n2. Analyze comments\n3. Summarize comments\n4. Exit")
  try:
    if(choice == 1) :
      model_directory = input("Please enter the location where to save the trained model : ")
      sample_size = int(input("Please enter the size of dataset to be used to train : "))
      epochs = int(input("Please enter the number of epochs : "))
      train_size = int(input("Please enter the ratio of data to be used to train the model : "))
      fineTune(model_directory, sample_size, epochs, train_size)
    elif(choice == 2) :
      API_Key = input("Please enter your YouTube Data API v3 Key : ")
      video_id = input("Please enter your YouTube Video ID : ")
      model_directory = input("Please enter the location of the trained model : ")
      analyzeComments(API_Key, video_id, model_directory)
    elif(choice == 3) :
      API_Key = input("Please enter your YouTube Data API v3 Key : ")
      video_id = input("Please enter your YouTube Video ID : ")
      summarizeComments(API_Key, video_id)
    elif(choice == 4) :
      print("Goodbye, see you later!")
      break
    else :
      print("Seems like an invalid choice!")
  except:
    print("Could you please check the inputs")
