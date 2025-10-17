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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TripletEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TripletEmbeddingNet, self).__init__()
        base_model = models.resnet18(pretrained=True)
        
        # Remove the final classification layer
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Identity()
        
        self.backbone = base_model
        self.embedding = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Optional: normalize
        return embeddings
    
    

# Define Classifier on top of Triplet Embedding Model

class TripletNetWithClassifier(nn.Module):
    def __init__(self, base_model, embedding_dim=128, num_classes=2, fine_tune=False):
        super().__init__()
        self.embedding_model = base_model
        
        # Option to fine-tune or freeze embedding model
        for param in self.embedding_model.parameters():
            param.requires_grad = fine_tune
        
        # Improved classifier head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            embedding = self.embedding_model(x)
        logits = self.classifier(embedding)
        return logits


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2, fine_tune=False):
        super(ResNet18Classifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)

        # Optionally freeze all layers
        if not fine_tune:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer for binary classification
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
    
    
class ResNet18ClassifierTuned(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18ClassifierTuned, self).__init__()
        self.base_model = models.resnet18(pretrained=True)

        # Freeze all layers first
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze layer3 and layer4 for fine-tuning
        for name, child in self.base_model.named_children():
            if name in ['layer4']:
                for param in child.parameters():
                    param.requires_grad = True

        # Replace the classifier head
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


