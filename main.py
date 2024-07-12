import torch
import torch.nn as nn # neural network module
import torch.optim as optim # optimizer module
from torch.utils.data import DataLoader, Dataset # dataset and dataloader utilities
import torchvision.transforms as transforms # torchvision (computer vision)
import torchaudio # torchaudio (audio processing)
from transformers import BertTokenizer, BertForSequenceClassification
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

# STEP ONE: Preprocess YouTube Comments

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

processed_comments = [preprocess_text(comment) for comment in comments]

# STEP TWO: Prepare the dataset

# label data
data = {
    "comment": processed_comments,
    "sentiment": [1, 0, -1]  # 1: positive, 0: neutral, -1: negative
}

# assign data to two-dimensional, size-mutable, tabular data
df = pd.DataFrame(data)

# split data into training data and testing/validating data
# this line makes 20% of the dataframe testing data and the rest is reserved
# for testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=19)
# this line takes another 20% from train_df and makes it validating data
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=19)

