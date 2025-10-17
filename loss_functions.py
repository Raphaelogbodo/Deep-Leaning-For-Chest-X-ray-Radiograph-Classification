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

# Loss functions 

# VERSION 1
class TripletLoss_v1(nn.Module):
    """
    Variant using squared Euclidean distances and optional L2‐normalization.
    Loss = mean( clamp( ‖a−p‖² − α‖a−n‖² + margin, 0 ) )
    """
    def __init__(self, margin: float = 1.0, alpha: float = 1.0, normalize: bool = False):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # Optionally normalize embeddings to unit length
        if self.normalize:
            anchor   = F.normalize(anchor,   p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)

        # Compute squared distances directly
        d_sq_ap = (anchor - positive).pow(2).sum(dim=1)
        d_sq_an = (anchor - negative).pow(2).sum(dim=1)

        # Hinge on squared‐distance margin
        loss = torch.clamp(d_sq_ap - self.alpha * d_sq_an + self.margin,
                           min=0.0)
        return loss.mean()



# VERSION 2 
class TripletLoss_v2(nn.Module):
    """
    Soft‐margin triplet + center loss (Huang et al. 2020 “Mor” style).
    Loss = mean(log(1 + exp[α(d_ap − d_an)])) + λ_c * MSE(a, center[label])
    """
    def __init__(self, alpha: float = 1.0, lambda_c: float = 0.01, embedding_dim: int = 128, num_classes: int = 2, normalize: bool = False):
        super().__init__()
        self.alpha    = alpha
        self.lambda_c = lambda_c
        self.normalize = normalize
        # One learnable center per class
        self.class_centers = nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        ).to(device)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Ensure labels are long for indexing
        labels = labels.long()

        # Optionally normalize embeddings and centers
        if self.normalize:
            anchor   = F.normalize(anchor,   p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
            centers  = F.normalize(self.class_centers, p=2, dim=1)
        else:
            centers = self.class_centers

        # Soft‐margin triplet term
        d_ap = F.pairwise_distance(anchor, positive, p=2)
        d_an = F.pairwise_distance(anchor, negative, p=2)
        triplet_loss = torch.log1p(torch.exp(self.alpha * (d_ap - d_an))).mean()

        # Center‐loss term
        centers_batch = centers[labels]
        center_loss = F.mse_loss(anchor, centers_batch)

        return triplet_loss + self.lambda_c * center_loss
