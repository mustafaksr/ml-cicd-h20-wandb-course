import torch
from sklearn.metrics import log_loss
from transformers import pipeline
import utils

def test_metrics():
    # Assuming logits are obtained from the transformer model
    

    # Initialize probabilities
    probabilities = None
    samples_ = 10000
    # Keep creating probabilities until the correct number of classes is achieved
    while probabilities is None or len(set(probabilities.argmax(axis=1).tolist())) != num_classes:
        logits = torch.randn(samples_, num_classes).cpu()  # Example: 1000 samples, num_classes output logits
        targets = torch.randint(0, num_classes, (samples_,)).cpu()  # Example: 1000 samples, random integer labels
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
