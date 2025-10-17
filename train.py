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
import shutil
from pathlib import Path

from model import *
from prepare_dataset import *
from loss_functions import *
from evaluate_embedding import *
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

num_epochs = 100


def start_training():
    device = logging_and_device()
    print(f"\nUsing Device ----> {device}\n")
    logging.info(f"Starting Deep Learning Models Based on Pretrained RESNET18; Number of Epochs ----> {num_epochs}\n")
    print(f"Starting Deep Learning Models Based on Pretrained RESNET18; Number of Epochs ----> {num_epochs}\n")
    train_val_test_data()
    train_embeddings()
    eval_all_embeddings()
    train_all_embed_classifiers()
    train_basemodels()


if os.path.exists(os.path.join(file_paths,'__pycache__')):
    shutil.rmtree(os.path.join(file_paths,'__pycache__'))

if os.path.exists(os.path.join(file_paths,'logs.log')):
    os.remove(os.path.join(file_paths,'logs.log'))

start_training()