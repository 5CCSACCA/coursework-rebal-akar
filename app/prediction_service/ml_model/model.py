#ml_model/model.py
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from typing import List, Dict
from pathlib import Path

current_dir = Path(__file__).parent
# Load model and tokenizer once at startup
model_path = current_dir / 'model' / 'data' / 'model.pth'
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model=torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# Label mapping
label_mapping = {
    0: "Neutral",
    1: "Offensive",
    2: "Hate"
}

async def predict_text(texts: List[str]) -> List[Dict]:
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
        probabilities = probabilities.cpu().numpy()

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
