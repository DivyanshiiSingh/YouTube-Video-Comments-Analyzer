import pandas as pd
from Fine_tune_mbert import fineTune
from Analyze_youtube_comments import analyzeComments
from Summarize_youtube_comments import summarizeComments

while True:
  choice = int(input("Hello! What are you intersted in ?\n1. Fine tune mBERT\n2. Analyze comments\n3. Summarize comments\n4. Exit"))
  try:
    if(choice == 1) :
      model_directory = input("Please enter the location where to save the trained model : ")
      epochs = int(input("Please enter the number of epochs : "))
      train_size = float(input("Please enter the ratio of data to be used to train the model : "))

      custom_df = None
      custom_df_path = input("Enter CSV file path for custom dataset (or press Enter to skip): ")
      if custom_df_path :
        try :
          custom_df = pd.read_csv(custom_df_path)
        except Exception as e :
          print(f"Error loading CSV : {e}\nSkipping custom dataset.")

      sample_size_input = input("Enter dataset sample size (or press Enter to use full dataset): ")
      sample_size = int(sample_size_input) if sample_size_input else None
      fineTune(model_directory, epochs, train_size, custom_df, sample_size)
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
    else:
      print("Invalid choice. Please enter a number between 1 and 4.")
  except Exception as e:
      print(f"Error occurred: {e}")
