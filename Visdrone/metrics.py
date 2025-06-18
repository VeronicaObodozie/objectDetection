
from utils import *
from tools import *
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils import data

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from torchvision import *

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, confusion_matrix

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns


#------------------- Template Metrics ------------------------#
def metrics_eval(model, loader, interp):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            image, label,_,_ = batch
            labels = label.squeeze().numpy()
            images = image.float().cuda()
            outputs = model(images)
           
            _, predicted = torch.max(outputs.data, 1)

            _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
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