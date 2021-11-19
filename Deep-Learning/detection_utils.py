from numpy.core.numeric import Inf
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
import os, sys
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from visualize import *


from torch.utils.data import Dataset
import torchvision

from torchvision import transforms
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN

from torch.utils.data.sampler import SubsetRandomSampler
import random

from torchvision.transforms import functional as F

# TODO: update the dataset path
full_dataset_path = "/Users/AlexandrosTzikas/Desktop/hw3/vkitti_2.0.3_rgb/{scene}/clone/frames/{type}/Camera_0/"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

###### Helper function for bounding box
def bounding_box(x):
    pos = np.where(x)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return (xmin, ymin), (xmax, ymax)

# DATASET CLASS

# Define dataset class
class VkittiDataset(Dataset):
    def __init__(self, full_dataset_path, transforms=None):
        self.root = full_dataset_path
        self.transforms = transforms
        
        # load all image files, sorting them to
        # ensure that they are aligned
        scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
        
        self.imgs = []
        for scene in scenes:
            dataset_path = full_dataset_path.format(scene=scene, type="rgb")
            for path in os.listdir(dataset_path):
                full_path = os.path.join(dataset_path, path)
                self.imgs.append(full_path)
        self.imgs = list(sorted(self.imgs))
        
        self.masks = []
        for scene in scenes:
            dataset_path = full_dataset_path.format(scene=scene, type="instanceSegmentation")
            for path in os.listdir(dataset_path):
                full_path = os.path.join(dataset_path, path)
                self.masks.append(full_path)
        self.masks = list(sorted(self.masks))
        
    def __getitem__(self, idx):
        
        img = Image.open(self.imgs[idx]).convert("RGB")    # open image and convert to RGB if grayscale
        mask = Image.open(self.masks[idx])   # open mask (no conversion)
    

        mask = np.array(mask)   # convert mask to np array. Mask has size (375, 1242) as the image 
        
        #TODO: find all unique entities present in the image. Remove any unwanted classes, such as 0 (background) (ok)
        obj_ids = np.sort(np.unique(mask))

        if obj_ids[0]==0:
            obj_ids = np.delete(obj_ids, 0)

        num_objs = len(obj_ids)

        #TODO: for each object id, create a corresponding binary mask
        # Each mask must be of type bool and true where the object is present and false everywhere else (ok)
        masks = np.zeros((num_objs, mask.shape[0], mask.shape[1]))  
        for i in range(num_objs):
            masks[i,:,:] = (mask==obj_ids[i])
    

        # get bounding box coordinates from the masks
        boxes = []
        issmall = np.zeros(num_objs, dtype=bool) # 1 if bounding box is smaller than 20X20 pixels
        
        # TODO: get bounding box coordinates from the masks
        for i in range(num_objs):
            # horizontal_indicies = np.where(np.any(masks[i,:,:], axis=0))[0]
            # vertical_indicies = np.where(np.any(masks[i,:,:], axis=1))[0]
            # if horizontal_indicies.shape[0]:
            #     x1, x2 = horizontal_indicies[[0, -1]]
            #     y1, y2 = vertical_indicies[[0, -1]]
            # else:
            #     x1, x2, y1, y2 = 0, 0, 0, 0
            boxes.append(np.array(bounding_box(masks[i,:,:])).flatten()) # boxes is in the form [num_objects,2,2]
        
            
        # TODO: remove masks and bounding boxes that are too small (ok)
        i = 0
        while i<len(obj_ids):
            if (boxes[i][2]-boxes[i][0]+1)*(boxes[i][3]-boxes[i][1]+1)<400:
                obj_ids = np.delete(obj_ids, i)
                boxes.pop(i)
                masks = np.delete(masks, i, axis=0)
            else: 
                i +=1
        
        num_objs = len(masks)

        target = {}
        target["boxes"] = boxes
        target["labels"] = obj_ids
        target["masks"] = masks
        target["num_objs"] = num_objs

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)


data = VkittiDataset(full_dataset_path)
img, target = data.__getitem__(2)


# CUSTOM TRANSFORMS

class CustRandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        # TODO: convert image and target to tensors (ok)
        # F.to_tensor converts input to tensor
        image = F.to_tensor(image)
        
        target['num_objs'] = torch.tensor([target['num_objs']])
        target['boxes'] = torch.from_numpy(np.array(target['boxes']).astype(np.float32))
        target['labels'] = torch.ones((target['num_objs'],), dtype=torch.int64)
        target['masks'] = torch.from_numpy(target['masks'].astype(np.uint8))
       
        ########################################################       
        # Previous implementation
        # t = []
        # items = []
        # for item in target:
        #     items.append(item)
        #     if item != 'labels' and item != 'num_objs':
        #         tens = F.to_tensor(np.array(target[item]))
        #     else: 
        #         tens = target[item]
        #     t.append(tens)
        
        # target = []
        # i = 0
        # for item in t:
        #     target.append({items[i]: item})
        #     i += 1
        ########################################################
        
        # TODO: With a probability of 0.5, flip the image, bounding box
        # and mask horizontally. (ok)
        # Hint: x.flip(axis) flips the tensor x along the provided axis 
        p = random.random()
        if p < self.prob:
            image = image.flip(2)
            for i in range(len(target['boxes'])):
                target['masks'][i] =  target['masks'][i].flip(1)
                target['boxes'][i]=torch.tensor(np.array(bounding_box(np.array(target['masks'][i]))).flatten())
                
        return image, target

randFlip = CustRandomHorizontalFlip(0.5)

print(1)
print(target['masks'][0,:])

img, target = randFlip.__call__(img, target)
print(2)
print(target['masks'][0,:])


# DATALOADER

def load_data(full_dataset_path):
    train_batch_size = 8
    test_batch_size = 1
    num_workers = 0
    test_fraction = 0.1

    transform = CustRandomHorizontalFlip(0.5)

    dataset = VkittiDataset(full_dataset_path, transforms=transform)

    # Function to convert batch of outputs to lists instead of tensors
    # in order to support variable sized images
    def collate_fn(batch):
        batch = filter (lambda x:x[1]["num_objs"] > 0, batch)
        return tuple(zip(*batch))

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(test_fraction * num_train))
    # TODO: Load the classification dataset into train and test loaders
    np.random.shuffle(indices)

    train_idx = indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)

    test_idx = indices[:split]
    test_sampler = SubsetRandomSampler(test_idx)

    #print(len(train_idx))
    #print(len(test_idx))
    print(1)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size = train_batch_size, sampler = train_sampler, num_workers = num_workers, collate_fn = collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size = test_batch_size, sampler = test_sampler, num_workers = num_workers, collate_fn = collate_fn
    )

    print(1)
    return (train_loader, test_loader)


train_loader, test_loader = load_data(full_dataset_path)


# MODEL 

def build_maskrcnn():
    # TODO: build and return a Mask R-CNN model with pretrained mobilenet v2 backbone  
    backbone = torchvision.models.mobilenet_v2(pretrained = True).features
    backbone.out_channels = 1280
    sizes = ((8,16,32,64,128),)
    aspect_ratios = ((0.5, 0.7, 1.2),)
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
    #anchor_generator.generate_anchors(tuple(sizes), tuple(aspect_ratios))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                    output_size=7,
                                    sampling_ratio=2)
    model = MaskRCNN(backbone,
                    num_classes=2,
                    rpn_anchor_generator = anchor_generator,
                    box_roi_pool = roi_pooler)

    return model

model = build_maskrcnn()
numOfParams = 0
for parameter in model.parameters():
    numOfParams += len(parameter)
print(numOfParams)


# TRAIN

def train(model, train_loader, optimizer, schedule, num_epochs):
    model.train()
    # TODO: Train the Mask R-CNN model

    for e in range(1, num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        for (X_batch, y_batch) in train_loader:
            #print(X_batch)
            #print(y_batch)
            #X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            loss_dict = model(X_batch, y_batch)

            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()
            schedule.step()
            epoch_loss += losses


        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')



optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
#train(model, train_loader, optimizer, scheduler, 2)


# EVALUATE

def eval(model, test_loader):
    model.eval()
    # TODO: Generate visualization of output on a batch of test images
    for X_batch, y_batch in test_loader:
        # TODO: compute the overall accuracy, confusion matrix,
        # precision and recall on the test dataset
        predictions = model(X_batch)
        display_images_wrapper(X_batch[0], predictions)
        break
eval(model, test_loader)