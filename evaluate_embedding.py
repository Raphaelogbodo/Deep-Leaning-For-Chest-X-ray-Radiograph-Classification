import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import os
from PIL import Image
from collections import Counter
import random
import kagglehub
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

from model import *
from prepare_dataset import *
from loss_functions import *
from training_functions import *

#file_paths=f"/Users/rnogbodo/deep-learning-project"
#model_path = f"/Users/rnogbodo/deep-learning-project/saved_model"
#loss_path = f"/Users/rnogbodo/deep-learning-project/loss"
#figures = f"/Users/rnogbodo/deep-learning-project/figures"

# File Paths
root_path = Path('.').resolve()
file_paths=f"{root_path}"
model_path = f"{root_path}/saved_model"
loss_path = f"{root_path}/loss"
figures = f"{root_path}/figures"

os.makedirs(model_path, exist_ok=True)
os.makedirs(loss_path, exist_ok=True)
os.makedirs(figures, exist_ok=True)

# Modified function to include cross-validation for different k values
def evaluate_embeddings_with_knn(model, train_loader, val_loader, device=device, k_values=[3, 5], n_splits=5):
    model.eval()
    
    def extract_embeddings(loader):
        embeddings = []
        labels = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                emb = model(imgs).cpu()
                embeddings.append(emb)
                labels.append(lbls)
        return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()

    # Extract train and validation embeddings and labels
    train_embeddings, train_labels = extract_embeddings(train_loader)
    val_embeddings, val_labels = extract_embeddings(val_loader)
    
    # Cross-validation loop to evaluate different k values
    def cross_validate_knn(X_train, y_train, k_values, n_splits):
        best_k = None
        best_accuracy = 0
        results = {}

        # Initialize StratifiedKFold for cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for k in k_values:
            fold_accuracies = []

            # Iterate over each fold
            for train_index, val_index in skf.split(X_train, y_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                # Initialize KNN classifier with the current k value
                knn = KNeighborsClassifier(n_neighbors=k)

                # Fit the model on the training fold
                knn.fit(X_train_fold, y_train_fold)

                # Make predictions on the validation fold
                y_pred = knn.predict(X_val_fold)

                # Calculate accuracy
                accuracy = accuracy_score(y_val_fold, y_pred)
                fold_accuracies.append(accuracy)

            # Compute average accuracy for the current k
            avg_accuracy = np.mean(fold_accuracies)
            results[k] = avg_accuracy

            # Check if this k gives better accuracy
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_k = k

        # Return the best k value and results
        return best_k, results

    # Perform cross-validation for KNN
    best_k, results = cross_validate_knn(train_embeddings, train_labels, k_values, n_splits)

    # Now fit the KNN classifier with the best k found from cross-validation
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_embeddings, train_labels)
    pred_labels = knn.predict(val_embeddings)

    # Print the classification report and accuracy for the best k
    print(f"\nBest k: {best_k}")
    logging.info(f"\nBest k: {best_k}")
    print("KNN Classification Report:")
    logging.info("KNN Classification Report:")
    print(classification_report(val_labels, pred_labels, target_names=["NORMAL", "PNEUMONIA"]))
    logging.info(classification_report(val_labels, pred_labels, target_names=["NORMAL", "PNEUMONIA"]))
    print(f"Accuracy: {accuracy_score(val_labels, pred_labels):.4f}\n")
    logging.info(f"Accuracy: {accuracy_score(val_labels, pred_labels):.4f}\n")

    return best_k, results


def plot_embeddings(model, loader, method="tsne", device=device,loss_v=None):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            emb = model(imgs).cpu()
            embeddings.append(emb)
            labels.append(lbls)
    
    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    elif method.lower() == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 4))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label="NORMAL" if label==0 else "PNEUMONIA", alpha=0.6)
    
    plt.title(f'{method.upper()} visualization of embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures, f'embed_loss_{loss_v}.pdf'), dpi=600)
    #plt.show()
    


