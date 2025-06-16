""" 
Created by Veronica Obodozie
Base AI Dev/Test Code
"""
#-----------------Importing Important Functions and Modules---------------------------#
#Functions
from utils import *
from metrics import *
import torch 
from torchvision.transforms import v2
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
import argparse
import os
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm


# Pennylane
import pennylane as qml
from pennylane import numpy as np
#-------------------------------------------------------------------------------#


#---------------------------------Data Preprocessing and loading----------------------------------------------#
def data_load(batch_size, num_workers):
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
    speed_root = "./speedplusv2/"
    train_set = Dataset('train', speed_root, data_transforms)
    val_set = Dataset('validation',  speed_root, data_transforms)
    test_set = Dataset('sunlamp',  speed_root, data_transforms)

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
    n_qubits = 7
    dev = qml.device("default.qubit", wires=n_qubits)

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    print(dev)
    #--------------------------------------------#
    # Set the hyperparameters
    print('------- Setting Hyperparameters-----------')
    batch_size = 16 # Change Batch Size o
    learning_rate = 1e-3 #4
    num_workers =2#4
    nepochs = 5 #"Use it to change iterations"
    weight_decay = 1e-4
    best_loss = 1e+20 # number gotten from initial resnet18 run
    stop_count = 7
    print(f'batch_size = {batch_size}, learning_rate = {learning_rate} num_workers = {num_workers} , nepochs = {nepochs} , best_loss = {best_loss}, weight_decay={weight_decay}')

    # Call Data Loader
    train_set, val_set, test_set, train_dataloader, val_dataloader, test_dataloader = data_load()

    #------------------ Training Parameters --------------------------#
    print('-------------------Pretrained Model Type: RESNET 50------------------------')
    resnet_input = (3, 224, 224)
    net = unet(resnet_input)
    PATH = './pose_unet.pth' # Path to save the best model
    net.to(device)

    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)  # all params trained
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    # Training
    dev(nepochs, net, device, optimizer, criterion, scheduler, train_dataloader, stop_count, val_dataloader, PATH)
    #---------------------- Testing ----------------------#
    print('Testing \n')
    net = PoseResNetModel(resnet_input)
    net.load_state_dict(torch.load(PATH))
    evaluate(net, lightbox_dataloader, device)
#-------------------------------------------------------------------------------#


#----------------------------------------- REFERENCES ---------------------------------#

#-------------------------------------------------------------------------------#