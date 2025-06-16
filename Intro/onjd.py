"""
Author: Veronica Obodozie
Goal: Try using object detection
Dataset:
"""
#-----------------Importing Important Functions and Modules---------------------------#
#Functions
from models import *
from metrics import *
import torch 
from torchvision import transforms, models 
from torchvision.transforms import v2
from torch.utils.data import DataLoader 

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import numpy as np
import sys
import time
from PIL import Image
from datetime import datetime
import csv

from sklearn.metrics import mean_squared_error, accuracy_score

import os


# Pennylane
import pennylane as qml
from pennylane import numpy as np

#-------------------------------------------------------------------------#



#-------------------------------------------------------------------------#

#-------------------------------------------------------------------------#

#-------------------------------------------------------------------------#

def main():
    # Check if GPU is available
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    n_qubits = 7
    dev = qml.device("default.qubit", wires=n_qubits)

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    print(dev)
    #--------------------------------------------#


    ##-----------------------------------------------------------------------------------------------------------##
    # Set the hyperparameters
    print('------- Setting Hyperparameters-----------')
    batch_size = 16 # Change Batch Size o
    learning_rate = 1e-3 #4
    num_workers =2#4
    nepochs = 5 #"Use it to change iterations"
    weight_decay = 1e-4
    best_loss = 1e+20 # number gotten from initial resnet18 run
    stop_count = 6
    print(f'batch_size = {batch_size}, learning_rate = {learning_rate} num_workers = {num_workers} , nepochs = {nepochs} , best_loss = {best_loss}, weight_decay={weight_decay}')
    ##-----------------------------------------------------------------------------------------------------------##


    #--------------------- Data Loading and Pre-processing  -----------------------#
    # Processing to match pre-trained networks
    print('------- DATA PROCESSING --------')
    # ResNet50
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transforms = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        # ...
        v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
        # ...
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Loading training set, using 20% for validation
    speed_root = "./speedplusv2/"
    train_dataset = PyTorchSatellitePoseEstimationDataset('train', speed_root, data_transforms)
    Validation_set = PyTorchSatellitePoseEstimationDataset('validation',  speed_root, data_transforms)
    sunlamp_test_set = PyTorchSatellitePoseEstimationDataset('sunlamp',  speed_root, data_transforms)
    lightbox_test_set = PyTorchSatellitePoseEstimationDataset('lightbox',  speed_root, data_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_dataloader = DataLoader(Validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    sunlamp_dataloader = DataLoader(sunlamp_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    lightbox_dataloader = DataLoader(lightbox_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #--------------------------------------------#
    #------------------ Training Parameters --------------------------#
    # print('-------------------Pretrained Model Type: RESNET 50------------------------')
    # resnet_input = (3, 224, 224)
    # net = PoseResNetModel(resnet_input)
    # PATH = './pose_resNet.pth' # Path to save the best model
    #net.to(device)

    # print('-------------------Pretrained Model Type: RESNET 50 Quantum Classifier------------------------')
    # resnet_input = (3, 224, 224)
    # net = PoseQuanModel(resnet_input, n_qubits)
    # PATH = './pose_quanFinal.pth' # Path to save the best model



    print('------------------- CNN Custom ------------------------')
    resnet_input = (3, 224, 224)
    net = PoseNew(resnet_input)
    PATH = './pose_newq.pth' # Path to save the best model

    net.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)  # all params trained
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

#--------------------------------------------#
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
    plt.savefig("lossfunction.png") 
    plt.show()
    print('Finished Training\n')
    #--------------------------------------------#
    #---------------------- Testing ----------------------#
    print('Testing \n')
    # net = PoseResNetModel(resnet_input)
    # net = PoseQuanModel(resnet_input, n_qubits)
    net = objD(resnet_input)
    net.load_state_dict(torch.load(PATH))
    print('---------------LIGHTBOX TESTING----------------')
    evaluate(net, lightbox_dataloader, device)
    print('---------------SUNLAMP TESTING----------------')
    evaluate(net, sunlamp_dataloader, device)
    #--------------------------------------------#

if __name__ == '__main__':
    main()
#-------------------------------------------------------------------------#

#---------------------References-----------------------#
#Challenge: https://kelvins.esa.int/pose-estimation-2021/challenge/
#Code: 