
# custom python scripts
from utils import *

# General Packages
import numpy as np
import os
import cv2
import time
from datetime import datetime
import csv
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import pandas as pd
# xml library for parsing xml files
from xml.etree import ElementTree as et
# matplotlib for visualization
import matplotlib.patches as patches

# AI Packages
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import *
from torchvision import transforms as torchtrans 
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn as nn
import torch.nn.functional as F
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as Fv
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes

# Data processing
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes

# Models
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from sklearn.model_selection import KFold
from ultralytics import YOLO

# Metrics
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns

# Pennylane
import pennylane as qml
from pennylane import numpy as np


#------------------- Template Metrics ------------------------#
def metrics_eval(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            image, label,_,_ = batch
            labels = label.squeeze().numpy()
            images = image.float().to(device)
            outputs = model(images)
           
            _, predicted = torch.max(outputs.data, 1)

            pred = pred.squeeze().data.cpu().numpy()  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # append true and predicted labels
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())

                # calculate macro F1 score
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')

        # Mean Average Precision
        calculate_map(y_pred, y_true, iou_threshold=0.5)

        # F1 score, Precision, Recall
        print(classification_report(y_true,y_pred))

        # TP, FP, TN, FN
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        iou = tp/(tp+fp+fn) # intersection over union
        print(f'Intersection over Union is: {iou}')
        # calculate micro F1 score 
        f1_micro = f1_score(y_true, y_pred, average='micro')
        print(f'F1-SCORE of the netwwork is given ass micro: {f1_micro}, macro: {f1_macro}')
        
        # ACCURACY
        accuracy = 100*accuracy_score(y_true, y_pred, normalize=True)
        print(f'Accuracy of the network on the test images: {accuracy} %')
        
        # #CONFUSION MATRIX
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()




# visualize bounding boxes in the image
def plot_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(10,15)
    a.imshow(img)
    label_names = {0: '', 1: 'pedestrian', 2: 'people', 3 : 'bicycle', 4: 'car', 5: 'van', 6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor', 11:''}
    for box, label in zip(target['boxes'], target['labels']):
      x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
      rect = patches.Rectangle((x, y), width, height,linewidth=1, edgecolor='b', facecolor='none')
      a.add_patch(rect)
      label_text = label_names[label.item()]
      a.text(x, y, f"{label_text}", color='r', fontsize=8)
    plt.show()
# img, target = dataset_train[0]
# plot_bbox(img, target)

def calculate_map(prediction, target, iou_threshold=0.5):
    pred_labels = prediction['labels']
    true_labels = target['labels']
    matched_true_boxes = np.zeros(len(true_labels))
    matched_scores = np.zeros(len(pred_labels))
    for i, pred_box in enumerate(prediction['boxes']):
        for j, true_box in enumerate(target['boxes']):
            iou = calculate_iou(pred_box, true_box)
            if iou > iou_threshold and true_labels[j] == pred_labels[i]:
                matched_true_boxes[j] = 1
                matched_scores[i] = prediction['scores'][i]
    accuracy = np.sum(matched_true_boxes) / len(true_labels)
    return accuracy