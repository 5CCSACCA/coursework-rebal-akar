"""
Machine Learning Model Utilities

This module loads a pre-trained DistilBERT model for sequence classification
and provides functionality for making predictions on input texts.
"""
import torch
import logging
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger("prediction_service.ml_model.model")
current_dir = Path(__file__).parent
# Load model and tokenizer once at startup
model_path = current_dir / 'model' / 'data' / 'model.pth'
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model=torch.load(model_path, map_location=device)
model.to(device)
model.eval()

label_mapping = {
    0: "Neutral",
    1: "Offensive",
    2: "Hate"
}

async def predict_text(texts: List[str]) -> List[Dict]:
    """
    Predict the sentiment of input texts.

    Args:
        texts (List[str]): A list of input texts to analyze.

    Returns:
        List[Dict]: A list of prediction results, including probabilities and labels.

    Raises:
        Exception: If prediction fails due to an error.
    """
    
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            probabilities = probabilities.cpu().numpy()
        logger.info("Model prediction completed successfully.")
    except Exception as e:
        logger.exception(f"Error during model prediction: {e}")    
        raise e
       
    # Prepare the results
    results = []
    for idx in range(len(texts)):
        pred_class = int(predicted_classes[idx])
        result = {
            "input_text": texts[idx],
            "prediction": pred_class,
            "prediction_label": label_mapping[pred_class],
            "probabilities": probabilities[idx].tolist(),
        }
        results.append(result)
    return results
