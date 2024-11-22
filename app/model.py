import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, AdamW, get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import numpy as np

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define a custom PyTorch Dataset
class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Load tokenized data
def load_tokenized_data(tokenized_data_dir):
    train_encodings, train_labels = torch.load(os.path.join(tokenized_data_dir, 'train.pt'))
    val_encodings, val_labels = torch.load(os.path.join(tokenized_data_dir, 'val.pt'))
    test_encodings, test_labels = torch.load(os.path.join(tokenized_data_dir, 'test.pt'))

    train_dataset = HateSpeechDataset(train_encodings, train_labels)
    val_dataset = HateSpeechDataset(val_encodings, val_labels)
    test_dataset = HateSpeechDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":

    tokenized_data_dir = os.path.join('..', 'data', 'data_processed', 'tokenized_data')
    train_dataset, val_dataset, test_dataset = load_tokenized_data(tokenized_data_dir)
    
    print(f"Training Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

# Initialize the DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Initialize the DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Move the model to GPU if available, unavailable in current computer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

print(f"Using device: {device}")