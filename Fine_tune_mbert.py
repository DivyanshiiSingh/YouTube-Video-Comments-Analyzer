import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from evaluate import load
from Generate_mapped_sentiment_data import create_labelled_dataframe

def fineTune(model_directory, epochs, train_size_ratio, custom_df = None, sample_size = None):
  if custom_df is None:
    if sample_size is None:
      raise ValueError("Either custom_df or sample_size must be provided.")
    else:
      df = create_labelled_dataframe(sample_size)
      if df is None:
        print("Dataframe could not be formed. Please check sample size and try again")
        return
  else:
    df = custom_df

  label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
  labels = torch.tensor([label_dict[label] for label in df['sentiment']])

  tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

  tokenized_inputs = tokenizer(df['comment'].tolist(), padding=True, truncation=True, return_tensors="pt")
  # mBERT defaults to max_length=512, the longest possible sequence length, so did not specify it

  dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels)

  train_size = int(train_size_ratio * len(dataset))
  test_size = len(dataset) - train_size

  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, drop_last=True)
  test_dataloader = DataLoader(test_dataset, batch_size=16)

  model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)

  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
  num_training_steps = len(train_dataloader) * epochs
  lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.to(device)

  model.train()
  for epoch in range(epochs):
      print(f"Epoch {epoch + 1} starting...")
      total_loss = 0.0
      for batch in train_dataloader:
          batch = [item.to(device) for item in batch]
          inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

          outputs = model(**inputs)

          loss = outputs.loss
          loss.backward()
          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          total_loss += loss.item()
      print(f"Epoch {epoch + 1} loss: {(total_loss / len(train_dataloader)):.2f}")
      print(f"Epoch {epoch + 1} completed.\n")

  metric = load("accuracy")
  model.eval()
  for batch in test_dataloader:
      batch = [item.to(device) for item in batch]
      inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

      with torch.no_grad():
          outputs = model(**inputs)

      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)
      metric.add_batch(predictions=predictions, references=batch[2])

  accuracy = metric.compute()
  print(f"Test Accuracy: {accuracy['accuracy'] * 100:.2f}%")

  try:
      model.save_pretrained(model_directory)
      tokenizer.save_pretrained(model_directory)
      print("Model and tokenizer saved successfully.")
  except Exception as e:
      print(f"Error saving model/tokenizer: {e}")
      raise
