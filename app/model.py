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



def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Get predictions
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            label_ids = labels.cpu().numpy()

            # Store predictions and true labels
            predictions.extend(preds)
            true_labels.extend(label_ids)

    # Calculate metrics
    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted')

    return acc, precision, recall, f1


def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=3):
    mlflow.start_run()

    # Log parameters
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
    mlflow.log_param("batch_size", train_loader.batch_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(train_loader):
            # Move batch to device
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)

            # Zero gradients
            model.zero_grad()

            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Backward pass
            loss.backward()
            total_loss += loss.item()

            # Gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"Batch {step}/{len(train_loader)}, Loss: {loss.item()}")

        # Calculate average loss over the epoch
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Log average training loss
        mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)

        # Evaluate on validation set
        val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc}, F1 Score: {val_f1}")

        # Log validation metrics
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_precision", val_precision, step=epoch)
        mlflow.log_metric("val_recall", val_recall, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)

    # Evaluate on test set after training
    test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc}, F1 Score: {test_f1}")

    # Log test metrics
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)

    # Log the trained model
    mlflow.pytorch.log_model(model, "distilbert_hatespeech_model")

    mlflow.end_run()





if __name__ == "__main__":
    # Define paths
    tokenized_data_dir = os.path.join('..', 'data', 'data_processed', 'tokenized')

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_tokenized_data(tokenized_data_dir)

    print(f"Training Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    # Initialize tokenizer and model
    tokenizer, model = initialize_model(device)

    # Define DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define training parameters
    epochs = 3
    learning_rate = 2e-5
    epsilon = 1e-8

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Start training
    train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=epochs)


