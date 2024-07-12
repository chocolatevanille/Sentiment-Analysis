import torch
import torch.nn as nn # neural network module
import torch.optim as optim # optimizer module
from torch.utils.data import DataLoader, Dataset # dataset and dataloader utilities
import torchvision.transforms as transforms # torchvision (computer vision)
import torchaudio # torchaudio (audio processing)
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import requests
from dotenv import load_dotenv
import os
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as tokenize
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from sklearn.model_selection import train_test_split 
import numpy as np

# load environment variables from the .env file
load_dotenv()

# access the environment variables
api_key = os.getenv('API_KEY')

#sample data, replace with actual comments later
comments = [
    "I love this video! It's awesome!",
    "This is the worst video I've ever seen.",
    "It's okay, not great but not terrible either."
]

# STEP 1: PREPROCESS YOUTUBE COMMENTS

# preprocessing function
# str -> str
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text) # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A) # remove numbers and special characters
    text = text.lower()
    tokens = tokenize(text)
    stop_words = set(stopwords.words('english')) # get and remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

preprocessed_comments = [preprocess_text(comment) for comment in comments]

# STEP 2: PREPARE THE DATASET

# label data
data = {
    "comment": preprocessed_comments,
    "sentiment": [1, 0, 2]  # 1: positive, 0: neegative, 2: neutral
}

# assign data to two-dimensional, size-mutable, tabular data
df = pd.DataFrame(data)

# split data into training data and testing/validating data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=19)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=19)

# STEP 3: BUILD THE MODEL

# 3.1: CREATE A CUSTOM DATASET CLASS

# sample dataset class
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment
        self.targets = dataframe.sentiment
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

# 3.2: PREPARING DATA LOADERS

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 1e-5

# Initialize the Bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
train_dataset = SentimentDataset(train_df, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_df, tokenizer, MAX_LEN)
test_dataset = SentimentDataset(test_df, tokenizer, MAX_LEN)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3.3: INITIALIZE THE BERT MODEL

# Define model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

