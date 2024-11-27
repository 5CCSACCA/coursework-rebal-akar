import os
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import random
import numpy as np
import argparse
import mlflow
import mlflow.pytorch
import itertools
from torch.cuda.amp import GradScaler, autocast

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Custom PyTorch Dataset
class HateSpeechDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Convert each encoding into a dictionary of tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Load tokenized data from local directory
def load_tokenized_data(tokenized_data_dir='tokenized'):
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
    # Load pre-trained DistilBERT model for sequence classification
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3  # Adjust based on your classification task
    )
    model.to(device)
    print(f"Using device: {device}")
    return model

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels, probabilities = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs and labels to the device
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            label_ids = labels.cpu().numpy()
            probs = probs.detach().cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(label_ids)
            probabilities.extend(probs)

    # Calculate metrics
    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )

    # ROC-AUC for multiclass
    roc_auc = None
    if len(set(true_labels)) > 2:
        try:
            roc_auc = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='weighted')
        except ValueError:
            pass  # Handle case where ROC-AUC can't be calculated
    else:
        roc_auc = roc_auc_score(true_labels, [prob[1] for prob in probabilities])

    return acc, precision, recall, f1, roc_auc

# Training and Evaluation Function with Early Stopping
def train_and_evaluate(args, train_dataset, val_dataset, test_dataset, patience=3):
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = initialize_model(device)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_loader) // args.accumulation_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Initialize GradScaler for mixed-precision training
    scaler = GradScaler()

    best_f1 = 0.0
    best_model_state = None
    patience_counter = 0  # Initialize patience counter

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(train_loader):
            # Move inputs and labels to the device
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)

            with autocast():
                # Forward pass
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / args.accumulation_steps  # Normalize loss

            # Backward pass
            scaler.scale(loss).backward()
            total_loss += loss.item()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters every 'accumulation_steps' batches
            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Print loss every 100 steps
            if step % 100 == 0:
                print(f"Batch {step}/{len(train_loader)}, Loss: {loss.item() * args.accumulation_steps:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Log training loss
        mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)

        # Evaluate on validation set
        val_acc, val_precision, val_recall, val_f1, val_roc_auc = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")

        # Log validation metrics
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_precision", val_precision, step=epoch)
        mlflow.log_metric("val_recall", val_recall, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)
        if val_roc_auc is not None:
            mlflow.log_metric("val_roc_auc", val_roc_auc, step=epoch)

        # Early Stopping Logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict()
            print(f"New best model found at epoch {epoch + 1} with F1 Score: {best_f1:.4f}")
            torch.save(best_model_state, "best_model.pt")  # Save the best model
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            print(f"No improvement in F1 score for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break  # Exit the training loop

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Final Evaluation on Test Set
    test_acc, test_precision, test_recall, test_f1, test_roc_auc = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")

    # Log test metrics
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    if test_roc_auc is not None:
        mlflow.log_metric("test_roc_auc", test_roc_auc)

    # Save the best model with a unique name
    model_save_path = f"model_lr{args.learning_rate}_bs{args.batch_size}_epochs{args.epochs}_accum{args.accumulation_steps}.pt"
    torch.save(best_model_state, model_save_path)
    print(f"Best model saved to {model_save_path}")

    # Log the model to MLflow
    mlflow.pytorch.log_model(model, artifact_path="model")

    # End MLflow run
    mlflow.end_run()

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DistilBERT for Hate Speech Classification')
    parser.add_argument('--tokenized_data_dir', type=str, default='tokenized', help='Directory containing tokenized data')
    args = parser.parse_args()

    # Initialize MLflow experiment
    mlflow.set_experiment("DistilBERT_HateSpeech_Classification")

    # Load datasets once
    train_dataset, val_dataset, test_dataset = load_tokenized_data(tokenized_data_dir=args.tokenized_data_dir)
    print(f"Training Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    # Define hyperparameter grid
    learning_rates = [1e-5, 2e-5, 3e-5]
    batch_sizes = [8, 16]
    epochs_list = [3, 5]
    accumulation_steps_list = [1, 2]

    # Loop over hyperparameter combinations
    for lr, bs, epochs, accum_steps in itertools.product(learning_rates, batch_sizes, epochs_list, accumulation_steps_list):
        # Update args with current hyperparameters
        args.learning_rate = lr
        args.batch_size = bs
        args.epochs = epochs
        args.accumulation_steps = accum_steps

        # Start MLflow run
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", bs)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("accumulation_steps", accum_steps)

            print(f"\nStarting run with lr={lr}, batch_size={bs}, epochs={epochs}, accumulation_steps={accum_steps}")

            # Call the training and evaluation function with early stopping
            train_and_evaluate(args, train_dataset, val_dataset, test_dataset, patience=3)
