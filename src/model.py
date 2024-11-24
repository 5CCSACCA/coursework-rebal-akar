import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup
)
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import numpy as np
import argparse
from azureml.core import Run, Dataset

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Custom PyTorch Dataset
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

# Load tokenized data from Azure ML Dataset
def load_tokenized_data(tokenized_data_dir):
    try:
        train_encodings, train_labels = torch.load(os.path.join(tokenized_data_dir, 'train.pt'))
        val_encodings, val_labels = torch.load(os.path.join(tokenized_data_dir, 'val.pt'))
        test_encodings, test_labels = torch.load(os.path.join(tokenized_data_dir, 'test.pt'))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Tokenized data file not found: {e.filename}")

    train_dataset = HateSpeechDataset(train_encodings, train_labels)
    val_dataset = HateSpeechDataset(val_encodings, val_labels)
    test_dataset = HateSpeechDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset

# Initialize DistilBERT model and tokenizer
def initialize_model(device):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    model.to(device)
    print(f"Using device: {device}")
    return tokenizer, model

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            label_ids = labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(label_ids)

    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted')

    return acc, precision, recall, f1

# Training function with MLflow logging
def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=3, accumulation_steps=4):
        # Only the main process should start an MLflow run
    if dist.get_rank() == 0:
        mlflow.start_run()

        # Log hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("accumulation_steps", accumulation_steps)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        if dist.get_rank() == 0:
            print(f"\nEpoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(train_loader):
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / accumulation_steps
            logits = outputs.logits

            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0 and dist.get_rank() == 0:
                print(f"Batch {step}/{len(train_loader)}, Loss: {loss.item() * accumulation_steps}")

        avg_train_loss = total_loss / len(train_loader)
        if dist.get_rank() == 0:
            print(f"Average Training Loss: {avg_train_loss}")
            mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)

        # Evaluation
        val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        if dist.get_rank() == 0:
            print(f"Validation Accuracy: {val_acc}, F1 Score: {val_f1}")
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

    # Final Evaluation on Test Set
    test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
    if dist.get_rank() == 0:
        print(f"\nTest Accuracy: {test_acc}, F1 Score: {test_f1}")
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)

        # Log the trained model
        mlflow.pytorch.log_model(model, "distilbert_hatespeech_model")
        mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    args = parser.parse_args()

    #Initialize distributed environment
    dist.init_process_group(backend='gloo')  # Use 'gloo' for CPU, 'nccl' for GPU
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')


    # Retrieve environment variables
    storage_account_key = os.getenv('STORAGE_ACCOUNT_KEY')
    storage_account_conn_str = os.getenv('STORAGE_ACCOUNT_CONNECTION_STRING')
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    mlflow_s3_endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')

    # Validate environment variables
    if not all([storage_account_key, storage_account_conn_str, mlflow_tracking_uri, mlflow_s3_endpoint_url]):
        raise ValueError("Missing one or more environment variables: STORAGE_ACCOUNT_KEY, STORAGE_ACCOUNT_CONNECTION_STRING, MLFLOW_TRACKING_URI, MLFLOW_S3_ENDPOINT_URL")

    # Configure MLflow
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = mlflow_s3_endpoint_url
    os.environ['AWS_ACCESS_KEY_ID'] = storage_account_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = storage_account_conn_str

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])


    # Load datasets
    tokenized_data_dir = os.path.join('tokenized')
    train_dataset, val_dataset, test_dataset = load_tokenized_data(tokenized_data_dir)
    print(f"Training Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    tokenizer, model = initialize_model(device)
     # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # Configure PyTorch for multi-core processing
    torch.set_num_threads(4)

    # Define DataLoaders with DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # Define DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_loader) // args.accumulation_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Start training

    train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=args.epochs, accumulation_steps=args.accumulation_steps)
    dist.destroy_process_group()