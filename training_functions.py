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
import time
import logging
from pathlib import Path

from model import *
from prepare_dataset import *
from loss_functions import *
from evaluate_embedding import *


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

num_epochs = 100

def start_logging():
    logging.basicConfig(filename=os.path.join(file_paths, f'logs.log'), level=logging.INFO)

# Device configuration
def check_torch_validity():
    """
    Checks the validity of the Torch library by performing several operations.

    This function checks the availability of a CUDA device and sets the device to be
    used by Torch accordingly. It also prints the device being used by Torch, the
    version of Torch, and the version of CUDA used for Torch compilation. Additionally,
    it attempts to perform a CUDA operation using `torch.zeros(1).cuda()` and catches
    any potential runtime errors.

    Returns:
        device (torch.device): The device used by Torch, either "cuda" or "cpu".

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device used by torch: {device}')
    logging.info(f'Version of torch: {torch.__version__}')
    logging.info(f'CUDA used for torch compilation: {torch.version.cuda}')
    try:
        logging.info(f'{torch.zeros(1).cuda()}')
    except RuntimeError as inst:
        logging.info(f'Runtime Error {inst}')
    return device


model_path = f"/Users/rnogbodo/deep-learning-project/saved_model"
loss_path = f"/Users/rnogbodo/deep-learning-project/loss"
figures = f"/Users/rnogbodo/deep-learning-project/figures"


# Training loop for TripletEmbeddingNet

def train_TripletEmbeddingNet(model,triplet_train_loader,v_1=False,v_2=True,num_epochs=1):
    if v_1:
        criterion = TripletLoss_v1(margin=1.0, alpha=0.5)
    elif v_2:
        criterion = TripletLoss_v2(alpha=1.0, lambda_c=0.01, embedding_dim=128, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for anchor, positive, negative, labels in triplet_train_loader:
            anchor, positive, negative , labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)

            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            if v_1:
                loss = criterion(anchor_embed, positive_embed, negative_embed) # if TripletLoss_v1 loss function is used
            elif v_2:
                loss = criterion(anchor_embed, positive_embed, negative_embed, labels) # if TripletLoss_v2 loss function is used

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        avg_loss = running_loss/len(triplet_train_loader)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
    if v_1:
        torch.save(model.state_dict(), os.path.join(model_path,"TripletEmbeddingNet_with_loss_v_1.pth"))
        np.savetxt(os.path.join(loss_path,'train_loss_v_1.txt'), np.array(train_loss), fmt='%.4f')
    elif v_2:
        torch.save(model.state_dict(), os.path.join(model_path,"TripletEmbeddingNet_with_loss_v_2.pth"))
        np.savetxt(os.path.join(loss_path, 'train_loss_v_2.txt'), np.array(train_loss), fmt='%.4f')
    return 



# Training loop for Classifier Model

# ----------------------------
# Compute Class Weights for CrossEntropyLoss
# ----------------------------
def compute_class_weights(dataset):
    label_counts = Counter(dataset.targets)
    total = sum(label_counts.values())
    weights = [total / label_counts[i] for i in range(len(label_counts))]
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    return weights

# ----------------------------
# Train Classifier Head
# ----------------------------
def train_classifier_head(classifier_model, train_loader, val_loader, loss_v=None, epochs=10, lr=1e-3, device=device):
    classifier_model.to(device)
    
    # Compute class weights for balanced learning
    class_weights = compute_class_weights(train_loader.dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Only classifier head parameters are trainable
    optimizer = optim.Adam(classifier_model.classifier.parameters(), lr=lr)
    train_loss = []
    val_ac = []
    for epoch in range(epochs):
        classifier_model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = classifier_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        acc = validate_classifier(classifier_model, val_loader, device)
        val_ac.append(float(acc))
    torch.save(classifier_model.state_dict(), os.path.join(model_path, f"TripletNetWithClassifier_loss_{loss_v}.pth"))
    print(f"TripletNetWithClassifier_loss_{loss_v} model saved!\n")
    logging.info(f"TripletNetWithClassifier_loss_{loss_v} model saved!\n")
    np.savetxt(os.path.join(loss_path, f'train_classifier_loss_{loss_v}.txt'), np.array(train_loss), fmt='%.4f')
    np.savetxt(os.path.join(loss_path, f'val_classifier_accuracy_{loss_v}.txt'), np.array(val_ac), fmt='%.4f')
    return 


# ----------------------------
# Train Resnet18 base model with no fine-tune and fine-tuned layers 4
# ----------------------------
def train_resnet_basemodel(classifier_model, train_loader, val_loader, model_name=None, epochs=10, lr=1e-3, device=device):
    classifier_model.to(device)
    
    # Compute class weights for balanced learning
    class_weights = compute_class_weights(train_loader.dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Only classifier head parameters are trainable
    optimizer = optim.Adam(classifier_model.base_model.fc.parameters(), lr=lr)
    train_loss = []
    val_ac = []
    for epoch in range(epochs):
        classifier_model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = classifier_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        acc = validate_classifier(classifier_model, val_loader, device)
        val_ac.append(float(acc))
    torch.save(classifier_model.state_dict(), os.path.join(model_path, f"{model_name}.pth"))
    print(f"{model_name} model saved!\n")
    logging.info(f"{model_name} model saved!\n")
    np.savetxt(os.path.join(loss_path, f'train_{model_name}_loss.txt'), np.array(train_loss), fmt='%.4f')
    np.savetxt(os.path.join(loss_path, f'val_{model_name}_accuracy.txt'), np.array(val_ac), fmt='%.4f')
    return 

# ----------------------------
# Validation Function
# ----------------------------
def validate_classifier(model, loader, device=device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")
    logging.info(f"Validation Accuracy: {acc:.4f}")
    return f"{acc:.4f}"
    
    
# ----------------------------
# Test Classifier on Test Set
# ----------------------------
def test_classifier(model, test_loader, loss_v=None, device=device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    print("Test Set Classification Report:")
    logging.info("Test Set Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))
    logging.info(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["NORMAL", "PNEUMONIA"],
                yticklabels=["NORMAL", "PNEUMONIA"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(figures, f'conf_matrix_loss_{loss_v}.pdf'), dpi=600)
    #plt.show()


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
    print(f"Best k: {best_k}")
    logging.info(f"Best k: {best_k}")
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
    

# Download latest version
def prepare_all_dataset():
    path = kagglehub.dataset_download("assemelqirsh/chest-x-ray-dataset")
    file_path = os.path.join(path ,"chest_xray")
    # Load dataset
    TRAIN_DIR = os.path.join(file_path, "train")
    VAL_DIR = os.path.join(file_path, "val")
    TEST_DIR = os.path.join(file_path, "test")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset_train = PneumoniaDataset(TRAIN_DIR, transform=transform)
    dataset_val = PneumoniaDataset(VAL_DIR, transform=transform)
    dataset_test = PneumoniaDataset(TEST_DIR, transform=transform)

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    # === Create Triplet dataset ===
    triplet_train_dataset = TripletPneumoniaDataset(dataset_train)
    triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=32, shuffle=True)

def prepare_plot_dataset():
    path = kagglehub.dataset_download("assemelqirsh/chest-x-ray-dataset")
    file_path = os.path.join(path ,"chest_xray")
    # Load dataset
    TRAIN_DIR = os.path.join(file_path, "train")
    VAL_DIR = os.path.join(file_path, "val")
    TEST_DIR = os.path.join(file_path, "test")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset_train = PneumoniaDataset(TRAIN_DIR, transform=transform)
    dataset_val = PneumoniaDataset(VAL_DIR, transform=transform)
    dataset_test = PneumoniaDataset(TEST_DIR, transform=transform)

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    # === Create Triplet dataset ===
    triplet_train_dataset = TripletPneumoniaDataset(dataset_train)
    triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=32, shuffle=True)
    return dataset_train,dataset_val,dataset_test,train_loader,val_loader,test_loader, triplet_train_dataset,triplet_train_loader

path = kagglehub.dataset_download("assemelqirsh/chest-x-ray-dataset")
file_path = os.path.join(path ,"chest_xray")
# Load dataset
TRAIN_DIR = os.path.join(file_path, "train")
VAL_DIR = os.path.join(file_path, "val")
TEST_DIR = os.path.join(file_path, "test")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_train = PneumoniaDataset(TRAIN_DIR, transform=transform)
dataset_val = PneumoniaDataset(VAL_DIR, transform=transform)
dataset_test = PneumoniaDataset(TEST_DIR, transform=transform)

train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
# === Create Triplet dataset ===
triplet_train_dataset = TripletPneumoniaDataset(dataset_train)
triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=32, shuffle=True)

# Device configuration
def logging_and_device():
    start_logging()
    device = check_torch_validity()
    return device

# Prepare Datasets
def train_val_test_data():
    print(f"Preparing Dataset for training, validation, and testing\n")
    logging.info(f"Preparing Dataset for training, validation, and testing\n")
    prepare_all_dataset()


######  TRAINING EMBEDDINGS   ######
def train_embeddings():
    print(f"Training Embeddings Using Triplet-Loss version 1")
    logging.info(f"Training Embeddings Using Triplet-Loss version 1")
    model_embed = TripletEmbeddingNet(embedding_dim=128).to(device)
    start_time = time.time()
    train_TripletEmbeddingNet(model_embed, triplet_train_loader, v_1=True,  v_2=False, num_epochs=num_epochs)
    end_time = time.time()
    print(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")
    logging.info(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")


    print(f"Training Embeddings Using Triplet-Loss version 2")
    logging.info(f"Training Embeddings Using Triplet-Loss version 2")
    model_embed = TripletEmbeddingNet(embedding_dim=128).to(device)
    start_time = time.time()
    train_TripletEmbeddingNet(model_embed, triplet_train_loader, v_1=False, v_2=True,  num_epochs=num_epochs)
    end_time = time.time()
    print(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")
    logging.info(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")

###############   EVALUATING AND PLOTTING TRAINED EMBEDINGS
def eval_all_embeddings():
    # n_splits is the number of folds in the cross-validation, e.g., 5
    model_embed = TripletEmbeddingNet(embedding_dim=128).to(device)
    model_embed.load_state_dict(torch.load(os.path.join(model_path,'TripletEmbeddingNet_with_loss_v_1.pth')))
    print(f"Evaluating Embeddings Using Triplet-Loss version 1")
    logging.info(f"Evaluating Embeddings Using Triplet-Loss version 1")
    best_k, results = evaluate_embeddings_with_knn(model_embed, train_loader, val_loader, device=device, k_values=[3, 5], n_splits=5)
    print(f"Best k for KNNClassifier: {best_k}\nResults: {results}\n")
    logging.info(f"Best k for KNNClassifier: {best_k}\nResults: {results}\n")
    plot_embeddings(model_embed, val_loader, method="tsne", device=device, loss_v='v_1')


    # n_splits is the number of folds in the cross-validation, e.g., 5
    model_embed = TripletEmbeddingNet(embedding_dim=128).to(device)
    model_embed.load_state_dict(torch.load(os.path.join(model_path,'TripletEmbeddingNet_with_loss_v_2.pth')))

    print(f"Evaluating Embeddings Using Triplet-Loss version 2")
    logging.info(f"Evaluating Embeddings Using Triplet-Loss version 2")
    best_k, results = evaluate_embeddings_with_knn(model_embed, train_loader, val_loader, device=device, k_values=[3, 5], n_splits=5)
    print(f"Best k for KNNClassifier: {best_k}\nResults: {results}\n")
    logging.info(f"Best k for KNNClassifier: {best_k}\nResults: {results}\n")
    plot_embeddings(model_embed, val_loader, method="tsne", device=device, loss_v='v_2')

########   TRAINING CLASSIFIER    ####
def train_all_embed_classifiers():
    # Create model with classifier head
    print(f"Training Classifier Using Triplet-Loss version 1")
    logging.info(f"Training Classifier Using Triplet-Loss version 1")
    model_embed = TripletEmbeddingNet(embedding_dim=128).to(device)
    model_embed.load_state_dict(torch.load(os.path.join(model_path,"TripletEmbeddingNet_with_loss_v_1.pth")))
    classifier_model = TripletNetWithClassifier(model_embed, embedding_dim=128, num_classes=2, fine_tune=False)

    start_time = time.time()
    train_classifier_head(classifier_model, train_loader, val_loader, loss_v='v_1', epochs=num_epochs)
    end_time = time.time()
    print(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")
    logging.info(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")

    test_classifier(classifier_model, test_loader, loss_v='v_1', device=device)

    # Create model with classifier head
    print(f"Training Classifier Using Triplet-Loss version 2")
    logging.info(f"Training Classifier Using Triplet-Loss version 2")
    model_embed = TripletEmbeddingNet(embedding_dim=128).to(device)
    model_embed.load_state_dict(torch.load(os.path.join(model_path,"TripletEmbeddingNet_with_loss_v_2.pth")))
    classifier_model = TripletNetWithClassifier(model_embed, embedding_dim=128, num_classes=2, fine_tune=False)

    start_time = time.time()
    train_classifier_head(classifier_model, train_loader, val_loader, loss_v='v_2', epochs=num_epochs)
    end_time = time.time()
    print(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")
    logging.info(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")

    test_classifier(classifier_model, test_loader, loss_v='v_2', device=device)

# TRAINING ResNet18Classifier AND PartialFineTuneResNet18 (with normal Cross_entropy loss) for base line comparison with the Training Embedding on Triplet Loss
def train_basemodels():
    # No Fine Tune ######
    print(f"Training ResNet18Classifier with no fine-tuning\n")
    logging.info(f"Training ResNet18Classifier with no fine-tuning\n")
    model_resnet = ResNet18Classifier(num_classes=2).to(device)
    start_time = time.time()
    train_resnet_basemodel(model_resnet, train_loader, val_loader, model_name="ResNet18Classifier", epochs=num_epochs, lr=1e-3, device=device)
    test_classifier(model_resnet , test_loader, loss_v="ResNet18Classifier", device=device) # Here loss_v is model name
    end_time = time.time()
    print(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")
    logging.info(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")

    # With Fine Tune of Layer 4 ######
    print(f"Training ResNet18Classifier_tuned with fine-tuning of Layer 4\n")
    logging.info(f"Training ResNet18Classifier_tuned with fine-tuning of Layer 4\n")
    model_resnet_finetune = ResNet18ClassifierTuned(num_classes=2).to(device)
    start_time = time.time()
    train_resnet_basemodel(model_resnet_finetune, train_loader, val_loader, model_name="ResNet18ClassifierTuned", epochs=num_epochs, lr=1e-3, device=device)
    test_classifier(model_resnet_finetune, test_loader, loss_v="ResNet18ClassifierTuned", device=device) # Here loss_v is model name
    end_time = time.time()
    print(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")
    logging.info(f"Computational time -----> {(end_time - start_time):.2f} seconds\n")
