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

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, category in enumerate(["NORMAL", "PNEUMONIA"]):
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if not os.path.isdir(img_path):
                    self.image_paths.append(os.path.join(category_path, img_name))
                    self.labels.append(label)
                
        self.data = self.image_paths  
        self.targets = self.labels
        
    def __len__(self):
        return len(self.image_paths)
       
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
      
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img,label
    
# Triplet Dataset Wrapper
class TripletPneumoniaDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.label_to_indices = {0: [], 1: []}  # 0: NORMAL, 1: PNEUMONIA

        for idx, label in enumerate(base_dataset.labels):
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        anchor_img, anchor_label = self.base_dataset[idx]
        anchor_label = int(anchor_label.item())  # Convert tensor -> int

        # Positive sample (same class, different image)
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.label_to_indices[anchor_label])
        positive_img, _ = self.base_dataset[pos_idx]

        # Negative sample (different class)
        neg_label = 1 - anchor_label
        neg_idx = random.choice(self.label_to_indices[neg_label])
        negative_img, _ = self.base_dataset[neg_idx]

        return anchor_img, positive_img, negative_img, torch.tensor(anchor_label)