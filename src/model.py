import os
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import random
import numpy as np
import argparse
import itertools
import mlflow
import mlflow.pytorch
from torch.cuda.amp import GradScaler, autocast
from mlflow.tracking import MlflowClient


# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define class labels
CLASS_NAMES = {
    0: "neutral",
    1: "offensive",
    2: "hate_speech"
}

# Custom PyTorch Dataset
class HateSpeechDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # No need to wrap tensors again
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
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

# Initialize DistilBERT model
def initialize_model(device, num_labels=3):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
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
    
    # Calculate overall metrics
    acc = accuracy_score(true_labels, predictions)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, labels=[0, 1, 2], average=None
    )
    
    roc_auc = None
    if len(set(true_labels)) > 2:
        try:
            roc_auc = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='weighted')
        except ValueError:
            pass  # Handle case where ROC-AUC can't be calculated
    else:
        roc_auc = roc_auc_score(true_labels, [prob[1] for prob in probabilities])
    
    per_class_metrics = {}
    for idx, cls in CLASS_NAMES.items():
        per_class_metrics[f"precision_{cls}"] = precision[idx]
        per_class_metrics[f"recall_{cls}"] = recall[idx]
        per_class_metrics[f"f1_{cls}"] = f1[idx]
        per_class_metrics[f"support_{cls}"] = support[idx]
    
    return acc, per_class_metrics, roc_auc

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
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8, weight_decay=args.weight_decay)
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
        total_val_loss = 0
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

            # Update parameters every 'accumulation_steps' batches
            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
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
        val_acc, val_per_class_metrics, val_roc_auc = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Print per-class metrics
        for metric_name, metric_value in val_per_class_metrics.items():
            print(f"Validation {metric_name}: {metric_value:.4f}")

        # Calculate validation loss
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs, labels=labels)
                val_loss = outputs.loss
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        # Log validation loss and metrics
        mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        if val_roc_auc is not None:
            mlflow.log_metric("val_roc_auc", val_roc_auc, step=epoch)
        
        # Log per-class metrics to MLflow
        for metric_name, metric_value in val_per_class_metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)

        # Early Stopping Logic based on F1-score
        current_f1 = val_per_class_metrics.get("f1_hate_speech", 0.0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_state = model.state_dict()
            print(f"New best model found at epoch {epoch + 1} with F1 Score (hate_speech): {best_f1:.4f}")
            torch.save(best_model_state, "best_model.pt")  # Save the best model
            patience_counter = 0  
        else:
            patience_counter += 1
            print(f"No improvement in F1 score for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break  

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Final Evaluation on Test Set
    test_acc, test_per_class_metrics, test_roc_auc = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Print per-class metrics for test set
    for metric_name, metric_value in test_per_class_metrics.items():
        print(f"Test {metric_name}: {metric_value:.4f}")

    # Calculate test loss
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs, labels=labels)
            test_loss = outputs.loss
            total_test_loss += test_loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")

    # Log test metrics
    mlflow.log_metric("test_accuracy", test_acc)
    if test_roc_auc is not None:
        mlflow.log_metric("test_roc_auc", test_roc_auc)
    
    # Log per-class metrics for test set
    for metric_name, metric_value in test_per_class_metrics.items():
        mlflow.log_metric(metric_name.replace("test_", "test_"), metric_value)
    
    # Save the best model with a unique name
    model_save_path = f"model_lr{args.learning_rate}_wd{args.weight_decay}_bs{args.batch_size}_epochs{args.epochs}_accum{args.accumulation_steps}.pt"
    torch.save(best_model_state, model_save_path)
    print(f"Best model saved to {model_save_path}")

    # Log the model to MLflow
    mlflow.pytorch.log_model(model, artifact_path="model")


# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DistilBERT for Hate Speech Classification')
    parser.add_argument('--tokenized_data_dir', type=str, default='tokenized', help='Directory containing tokenized data')
    args = parser.parse_args()

    # Set the Tracking URI first
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.set_experiment("DistilBERT_HateSpeech_Classification")

    train_dataset, val_dataset, test_dataset = load_tokenized_data(tokenized_data_dir=args.tokenized_data_dir)
    print(f"Training Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    # Define hyperparameter grid
    learning_rates = [3e-5, 4e-5]
    weight_decays = [0.0, 0.01]
    batch_size = 9
    accumulation_steps = 2
    epochs = 9

    # Initialize variables to track the best overall run
    best_f1_overall = 0.0
    best_run_id = None

    # Initialize MLflow client for model registry
    client = MlflowClient()

    # Loop over hyperparameter combinations
    for lr, wd in itertools.product(learning_rates, weight_decays):
        # Update args with current hyperparameters
        args.learning_rate = lr
        args.weight_decay = wd
        args.batch_size = batch_size
        args.accumulation_steps = accumulation_steps
        args.epochs = epochs

        # Start MLflow run
        with mlflow.start_run() as run:
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("weight_decay", wd)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("accumulation_steps", accumulation_steps)

            print(f"\nStarting run with lr={lr}, weight_decay={wd}, batch_size={batch_size}, epochs={epochs}, accumulation_steps={accumulation_steps}")

            # Call the training and evaluation function with early stopping
            train_and_evaluate(args, train_dataset, val_dataset, test_dataset, patience=3)

            # After training, check if this run has the best F1 score
            run_info = run.info
            run_metrics = mlflow.get_run(run_info.run_id).data.metrics
            run_f1 = run_metrics.get("f1_hate_speech", 0.0)

            if run_f1 > best_f1_overall:
                best_f1_overall = run_f1
                best_run_id = run_info.run_id

    # After all runs, register the best overall model to Production
    if best_run_id:
        best_model_uri = f"runs:/{best_run_id}/model"
        model_name = "HateSpeechModel"

        try:
            # Register the model
            model_details = client.register_model(best_model_uri, model_name)
            print(f"Best Model registered: {model_details.name} version {model_details.version}")

            # Transition the best model to Production
            client.transition_model_version_stage(
                name=model_details.name,
                version=model_details.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"Best Model version {model_details.version} transitioned to Production")
        except Exception as e:
            print(f"Failed to register best model: {e}")
