import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset
from model import get_model

# Load dataset
df = pd.read_csv('data/dataset.csv')
df = df.dropna()

texts = (df['title'] + " " + df['text']).tolist()
labels = df['label'].tolist()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)

input_ids = torch.tensor(encodings['input_ids'])
attention_mask = torch.tensor(encodings['attention_mask'])
labels = torch.tensor(labels)

# Split
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids, labels, test_size=0.2
)

train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8)

# Model
model = get_model()
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
model.train()
for epoch in range(2):
    for batch in train_loader:
        inputs, labels = batch
        outputs = model(inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save model
torch.save(model.state_dict(), 'saved_model/model.pt')
