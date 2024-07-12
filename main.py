import torch
import torch.nn as nn # neural network module
import torch.optim as optim # optimizer module
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler # utilities
import torchvision.transforms as transforms # torchvision (computer vision)
import torchaudio # torchaudio (audio processing)
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW
import requests
from dotenv import load_dotenv
import os
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as tokenize
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# STEP 0: SET UP ENVIRONMENT

# load environment variables from the .env file
#load_dotenv()

# access the environment variables
#api_key = os.getenv('API_KEY')

# STEP 1: IMPORT AND PREPROCESS DATA

#training data
train_df_unfiltered = pd.read_json('Movies_and_TV.json', lines=True)
train_df = train_df_unfiltered.loc[:, ["overall", "reviewText"]]
#print(train_df.head())

# preprocessing functions
# cleans up text in reviews
# str -> str
def preprocess_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text) # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A) # remove numbers and special characters
    text = text.lower()
    tokens = tokenize(text)
    stop_words = set(stopwords.words('english')) # get and remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def shift_score(score):
    return (int(score) - 1)
    
train_df['reviewText'] = train_df['reviewText'].apply(preprocess_text)
train_df['overall'] = train_df['overall'].apply(shift_score) # need range 0-4, not 1-5
#print(train_df.head())

# STEP 2: TOKENIZE DATA FOR BERT

# initialize the Bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenize the text
max_len = 512
# tokenizer on every reviewText, special tokens to indicate CLS, SEP, etc. for bert
train_df['input_ids'] = train_df['reviewText'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True)) 
# post truncates sequences longer than and pads sequences shorter than max_len from the end 
input_ids_padded = pad_sequences(train_df['input_ids'].tolist(), maxlen=max_len, dtype="long", truncating="post", padding="post")
train_df['input_ids'] = input_ids_padded.tolist()
train_df['attention_masks'] = train_df['input_ids'].apply(lambda seq: [float(i > 0) for i in seq]) # set padding tokens to 0
#print(train_df[['overall', 'input_ids', 'attention_masks']].head())

# convert lists to PyTorch-specific tensors
input_ids = torch.tensor(train_df['input_ids'].values.tolist())
attention_masks = torch.tensor(train_df['attention_masks'].values.tolist())
labels = torch.tensor(train_df['overall'].values)

# create the DataLoader for training and validation sets
batch_size = 16

dataset = TensorDataset(input_ids, attention_masks, labels) # convert to fancy TensorDataset (exciting!!)
train_size = int(0.8 * len(dataset)) # change this to make more or less of set for training/validation
val_size = len(dataset) - train_size

# make the train and validation datasets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# prepare dataloaders for training and validation
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# STEP 3: INITIALIZE MODEL (BERT)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=5,  # number of unique sentiment classes
    output_attentions=False,
    output_hidden_states=False
)

# move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# STEP 4: TRAIN THE MODEL

learning_rate = 2e-5
epsilon = 1e-8

# optimize parameters
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

epochs = 4 # number of passes through the entire training dataset
total_steps = len(train_dataloader) * epochs

# scheduler initialization
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# training function
# NULL -> NULL
def train():
    model.train() # training mode (enables gradients and dropout)
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch_input_ids, batch_attention_mask, batch_labels = tuple(t.to(device) for t in batch)
        
        model.zero_grad() # clears gradient from previous batch
        
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels) # forward pass through model
        loss = outputs.loss
        total_loss += loss.item() # compute total loss
        
        loss.backward() # backwards propagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clips gradient to prevent exploding
        
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Training loss: {avg_train_loss}")

# validating function
# NULL -> NULL
def validate():
    model.eval() #evaluation mode (disables gradients and dropout)
    preds, true_labels = [], []
    
    for batch in val_dataloader:
        batch_input_ids, batch_attention_mask, batch_labels = tuple(t.to(device) for t in batch)

        # without computing gradients, perform forward pass
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)

        # get model predictions and true labels
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy()) 
        true_labels.extend(batch_labels.cpu().numpy())
    
    acc = accuracy_score(true_labels, preds)
    print(f"Validation Accuracy: {acc}")

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train()
    validate()

# this is taking forever to run... maybe decrease from 5000 reviews for training down to something more reasonable but less accurate