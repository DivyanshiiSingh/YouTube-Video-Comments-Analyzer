# training the model

import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from evaluate import load

df = create_labelled_dataframe(500)
if df.empty :
  print("DataFrame is empty, adjust the sample size")
  exit()


# converting the simplified labels to integers for model training
label_dict = {'positive': 2, 'negative': 0, 'neutral': 1}
labels = torch.tensor([label_dict[label] for label in df['sentiment']])
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenized_inputs = tokenizer(df['comment'].tolist(), padding="max_length", truncation=True, max_length=128, return_tensors="pt")

dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)  # Adjust batch size if needed
test_dataloader = DataLoader(test_dataset, batch_size=8)


model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 5  # For 5 epochs, can be adjusted as per dataframe size
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.train()
for epoch in range(5):
    print(f"Epoch {epoch + 1} training...")
    for batch in train_dataloader:
        batch = [item.to(device) for item in batch]
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1} completed.")

# Evaluation on test data
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

save_directory = '/path/where/to/save/Fine_tuned_mBERT'
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print("Model and tokenizer saved successfully.")
