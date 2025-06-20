# custom python scripts
from utils import *
from metrics import *

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

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')



#---------------------------------Data Preprocessing and loading----------------------------------------------#
def data_load(batch_size, num_workers):
    # Data location
    files_dir_train = '/content/VisDrone2019-DET-train/'
    files_dir_val = '/content/VisDrone2019-DET-val/'
    files_dir_test = '/content/VisDrone2019-DET-test-dev/'
    # Processing to match networks
    print('------- DATA PROCESSING --------')
    data_transforms = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        v2.Resize(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Loading training set, using 20% for validation
    train_set = VisDroneDataset(files_dir_train+'/images',files_dir_train+'/annotations', 1224, 724, transforms=data_transforms)
    val_set = VisDroneDataset(files_dir_val+'/images',files_dir_val+'/annotations', 1224, 724, transforms=data_transforms)
    test_set = VisDroneDataset(files_dir_test+'/images',files_dir_test+'/annotations', 1224, 724, transforms=data_transforms)
    # total_samples = len(train_set)

    # # Define split lengths
    # train_size = int(0.8 * total_samples)
    # val_size = total_samples - train_size # Remaining for test
    # # If using validation:
    # # val_size = int(0.1 * total_samples)
    # # train_size = total_samples - val_size - test_size

    # # Perform the split
    # train_dataset, val_set = random_split(train_set, [train_size, val_size])
    # # If using validation:
    # # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=utils.collate_fn)
    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_set, val_set, test_set, train_dataloader, val_dataloader, test_dataloader
#-------------------------------------------------------------------------------#

#--------------------------------- DEVELOPMENT ----------------------------------------------#
def dev(nepochs, net, device, optimizer, criterion, scheduler, train_dataloader, stop_count, val_dataloader, PATH):
    #---------------------- Training and Validation ----------------------#
    e = []
    trainL= []
    valL =[]
    counter = 0
    print('Traning and Validation \n')
    # Development time
    training_start_time = time.time()
    for epoch in range(nepochs):  # loop over the dataset multiple times
        # Training Loop
        net.train()
        train_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device).float()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/i
        print(f'{epoch + 1},  train loss: {train_loss :.3f},', end = ' ')
        scheduler.step()
    #---------------Validation----------------------------#
        net.eval()
        val_loss = 0.0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels] 
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_loss = val_loss/i
            print(f'val loss: {val_loss :.3f}')
                  
        valL.append(val_loss)
        trainL.append(train_loss)
        e.append(epoch)
        # Save best model
        if val_loss < best_loss:
            print("Saving model")
            torch.save(net.state_dict(), PATH)
            best_loss = val_loss
            counter = 0
            # Early stopping
        elif val_loss > best_loss:
            counter += 1
            if counter >= stop_count:
                print("Early stopping")
                break
        else:
            counter = 0
    print('Training finished, took {:.4f}s'.format(time.time() - training_start_time))
    # Total number of parameters
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f'Total Number of Parameters: {pytorch_total_params}')
    # Visualize training and Loss functions
    plt.figure()
    plt.plot(e, valL, label = "Val loss")
    plt.plot(e, trainL, label = "Train loss")
    plt.xlabel("Epoch (iteration)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("lossfunctionunet.png") 
    plt.show()
    print('Finished Training\n')
    #--------------------------------------------#

#-------------------------------------------------------------------------------#

#-------------------------------- MAIN FUNCTION -----------------------------------------------#
def main():
    # Check if GPU is available
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    # n_qubits = 7
    # dev = qml.device("default.qubit", wires=n_qubits)

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    print(dev)
    #--------------------------------------------#
    # Set the hyperparameters
    print('------- Setting Hyperparameters-----------')
    batch_size = 16 # Change Batch Size o
    learning_rate = 1e-3 #4 0.005
    num_workers =2#4
    num_classes = 12
    nepochs = 5 #"Use it to change iterations"
    weight_decay = 1e-4
    best_loss = 1e+20 # number gotten from initial resnet18 run
    stop_count = 7
    print(f'batch_size = {batch_size}, learning_rate = {learning_rate} num_workers = {num_workers} , nepochs = {nepochs} , best_loss = {best_loss}, weight_decay={weight_decay}')

    # Call Data Loader
    train_set, val_set, test_set, train_dataloader, val_dataloader, test_dataloader = data_load()

    # Helper functions for pytorch implementation
    os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
    os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
    os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
    os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
    os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")

    #------------------ Training Parameters --------------------------#
    print('-------------------Pretrained Model Type: RESNET 50------------------------')
    resnet_input = (3, 224, 224)
    net = base_MRCNN_model(num_classes)
    PATH = './pose_unet.pth' # Path to save the best model
    net.to(device)

    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)  # all params trained
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1, weight_decay=weight_decay)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    #------------- PYTORCH MODELS --------------#
    # Training
    dev(nepochs, net, device, optimizer, criterion, scheduler, train_dataloader, stop_count, val_dataloader, PATH)
    #---------------------- Testing ----------------------#
    print('Testing \n')
    net = base_MRCNN_model(num_classes)
    net.load_state_dict(torch.load(PATH))
    evaluate(net, test_dataloader, device)

    #---------- YOLO v11 --------------#
    # Visdrone
    # Load a pretrained model
    model = YOLO("yolo11n.pt")

    # Train the model
    results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640, name="yolo11nVisDrone")
    # Evaluate the model's performance on the validation set
    results = model.val()
    print('---------- EVALUATE model perfoemance on validation set ----------')
    print(results)
    # Export the model to ONNX format
    success = model.export()
#-------------------------------------------------------------------------------#

if __name__ == '__main__':
    main()

#----------------------------------------- REFERENCES ---------------------------------#

#-------------------------------------------------------------------------------#

