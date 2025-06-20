

# custom python scripts
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

# Data processing
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes

# Models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, resnet50, ResNet50_Weights
from sklearn.model_selection import KFold
from ultralytics import YOLO

# Metrics
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns

# Pennylane
import pennylane as qml
from pennylane import numpy as np


###------------------- DATASETS ------------------###
""" 
Datasets must return a tupple with a similar structure as:
1. image : with [3, H, W] # channels, height, width
2. target: dictionary
    - boxes, torchvision.tv_tensors.BoundingBoxes of shape [N, 4]: the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
    - labels, integer torch.Tensor of shape [N]: the label for each bounding box. 0 represents always the background class.
    - image_id, int: an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
    - area, float torch.Tensor of shape [N]: the area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
    - iscrowd, uint8 torch.Tensor of shape [N]: instances with iscrowd=True will be ignored during evaluation.
    - (optionally) masks, torchvision.tv_tensors.Mask of shape [N, H, W]: the segmentation masks for each one of the objects
 Requirements:
 pip install pycocotools
 [windows] pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
"""

#---------- PennFudan Dataset -----------#
## Ref https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html?highlight=maskrcnn_resnet50_fpn

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

#---------- VisDrone Dataset -----------#
## Ref: https://github.com/nia194/Object-Detection/blob/main/MaskRCNN/Visdrone_MaskRCNN.ipynb
class VisDroneDataset(torch.utils.data.Dataset):
    #image directory, annotation directory, width, height, transforms
    def __init__(self, img_dir,ann_dir, width, height, transforms=None):
        self.transforms = transforms
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.height = height
        self.width = width
        
        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(img_dir))
                        if image[-4:]=='.jpg']
        
        # classes: 0 index is reserved for background
        self.classes = [_, 'pedestrian','people','bicycle', 
                        'car','van','truck','tricycle','awning-tricycle','bus','motor']

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.img_dir, img_name)

        # reading the images and converting them to correct size and color    
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # dividing by 255
        img_res /= 255.0
        
        # annotation file
        annot_filename = img_name[:-4] + '.txt'
        annot_file_path = os.path.join(self.ann_dir, annot_filename)
        
        boxes = []
        labels = []
        
        with open(annot_file_path, 'r') as f:
            for line in f:
                box = [float(x) for x in line.strip().split(',')]
                labels.append(int(box[5]))
                
                xmin, ymin, w, h = box[:4]
                xmax = xmin + w
                ymax = ymin + h
                xmin_corr = (xmin/img.shape[1])*self.width
                xmax_corr = (xmax/img.shape[1])*self.width
                ymin_corr = (ymin/img.shape[0])*self.height
                ymax_corr = (ymax/img.shape[0])*self.height
                boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)    
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["masks"] = torch.zeros((boxes.shape[0], self.height, self.width), dtype=torch.uint8)
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:
            
            sample = self.transforms(image=img_res, bboxes=target['boxes'], labels=labels)
            
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
                      
        return img_res, target

    def __len__(self):
        return len(self.imgs)


#---------- Aerial Cars Dataset -----------#
## Ref: 

class AerialCarsDataset(torch.utils.data.Dataset):
    #image directory, annotation directory, width, height, transforms
    def __init__(self, list_path,  img_dir,ann_dir, width, height, transforms=None):
        self.transforms = transforms
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.height = height
        self.width = width
        self.list_path = list_path
        
        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [i_id.strip() for i_id in open(list_path)]
        # classes: 0 index is reserved for background
        self.classes = ['car', 'truck', 'bus', 'minibus', 'cyclist']

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.img_dir, img_name)

        # reading the images and converting them to correct size and color    
        # img = cv2.imread(image_path)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # # dividing by 255
        # img_res /= 255.0
        img = read_image(image_path)

        # annotation file
        annot_filename = img_name[:-4] + '.txt'
        annot_file_path = os.path.join(self.ann_dir, annot_filename)
        
        boxes = []
        labels = []
        
        with open(annot_file_path, 'r') as f:
            for line in f:
                box = [float(x) for x in line.strip().split(',')]
                labels.append(int(box[5]))
                
                xmin, ymin, w, h = box[:4]
                xmax = xmin + w
                ymax = ymin + h
                xmin_corr = (xmin/img.shape[1])*self.width
                xmax_corr = (xmax/img.shape[1])*self.width
                ymin_corr = (ymin/img.shape[0])*self.height
                ymax_corr = (ymax/img.shape[0])*self.height
                boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)    
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["masks"] = torch.zeros((boxes.shape[0], self.height, self.width), dtype=torch.uint8)
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:
            img, target = self.transforms(img, target)
            # sample = self.transforms(image=img_res, bboxes=target['boxes'], labels=labels)
            
            # img_res = sample['image']
            # target['boxes'] = torch.Tensor(sample['bboxes'])
                      
        return img, target #img_res

    def __len__(self):
        return len(self.imgs)



###------------------- MODELS ------------------###
# Finetuning model pretrained on COCO

# similar method in visdrone notebook and seeking finetuning with pytorch

def base_MRCNN_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

def base_MRCNN_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

# Modify pytorch base model
def mod_object_detection_model(num_classes):
    #--------- MODIFY BACKBONE AND ADD OUTPUT CHANNELS --------#
    # load a pre-trained model for classification and return only the features
    backbone = mobilenet_v2(weights="DEFAULT").features
    # ``FasterRCNN`` needs to know the number of output channels in a backbone. 
    backbone.out_channels = 1280

    #----------- MODIFY RPN (REGION PROPOSAL NETWORK) --------------#
    """ let's make the RPN generate 5 x 3 anchors per spatial
    location, with 5 different sizes and 3 different aspect
    ratios. We have a Tuple[Tuple[int]] because each feature
    map could potentially have different sizes and aspect ratios """
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    #-------------- ROI ALIGN -------------------#
    """ let's define what are the feature maps that we will use to perform the region of interest cropping, as well as
    the size of the crop after rescaling.  if your backbone returns a Tensor, featmap_names is expected to
    be [0]. More generally, the backbone should return an  ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    feature maps to use. """
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # put the pieces together inside a Faster-RCNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

def base_yolo(num_classes):
    # pretrained
    model = YOLO("yolo11n.pt")
    # Train from scratch
    model = YOLO("yolo11n.yaml")
    return model