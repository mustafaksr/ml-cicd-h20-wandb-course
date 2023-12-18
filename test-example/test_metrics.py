import torch
from sklearn.metrics import log_loss
from transformers import pipeline
import utils

def test_metrics():
    # Assuming logits are obtained from the transformer model
    logits = torch.randn(20, num_classes)  # Example: 20 samples, num_classes output logits
    targets = torch.randint(0, num_classes, (20,))  # Example: 20 samples, random integer labels

    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()

    # Compute log loss
    logloss = utils.compute_metrics((probabilities, targets))
    print(logloss)

    # Add assert statement
    assert logloss["log_loss"] >= 0.0  # You can modify this condition based on your expectations
    assert logloss["log_loss"] <= 5.0  # You can modify this condition based on your expectations
# Number of classes in your text classification task
num_classes = 7  # Change this to the actual number of classes in your task

if __name__ == "__main__":
    test_metrics()
