import torch
from transformers import BertTokenizer
from model import get_model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = get_model()
model.load_state_dict(torch.load('saved_model/model.pt'))
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    return "Real News" if prediction == 1 else "Fake News"
