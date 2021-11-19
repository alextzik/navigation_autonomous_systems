from numpy.core.numeric import Inf
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import pandas as pd
import os, sys

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class_dataset_path = "/Users/AlexandrosTzikas/Desktop/hw3/class_dataset/"

# DATALOADER

def load_data(class_dataset_path):
    batch_size = 8
    num_workers = 0
    test_fraction = 0.1

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.61717, 0.6252, 0.5192],
            std = [0.25209, 0.244, 0.2677]
        ),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    # TODO: Add transforms for preprocessing and data augmentation. 
    # You should atleast include: 
    # - resize to 32X32 images
    # - Normalize by the mean and std deviation of the dataset
    # - Randomly flip the image horizontally with a probability of 0.5

    dataset = datasets.ImageFolder(class_dataset_path, transform=transform)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(test_fraction * num_train))
    # TODO: Load the classification dataset into train and test loaders
    np.random.shuffle(indices)

    train_idx = indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)

    test_idx = indices[:split]
    test_sampler = SubsetRandomSampler(test_idx)

    print(len(train_idx))
    print(len(test_idx))

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size = batch_size, sampler = train_sampler, num_workers = num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size = batch_size, sampler = test_sampler, num_workers = num_workers
    )

    return (train_loader, test_loader)

# NEURAL NETWORK ARCHITECTURE

class VehicleClassifier(nn.Module):
    def __init__(self):
        super(VehicleClassifier, self).__init__()
        # TODO: create neural network layers for classification 
        self.conv1 = nn.Conv2d(3,6,5)
        self.fc1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6,16, 5)
        self.fc2 = nn.MaxPool2d(2)
        self.fc3 = nn.Linear(400,120)
        self.fc4 = nn.Linear(120,84)
        self.fc5 = nn.Linear(84,1)



    def forward(self, x):
        # TODO: write the forward pass of the neural network
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = self.fc1(x)
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = self.fc2(x)
        #print(x.size())
        x = torch.reshape(x, (x.size()[0], x.size()[1]*x.size()[2]*x.size()[3]))
        x = F.relu(self.fc3(x))
        #print(x.size())
        x = F.relu(self.fc4(x))
        #print(x.size())
        x = self.fc5(x)
        #print(x.size())
        x = F.sigmoid(x)

        return x

# Print network structure
net = VehicleClassifier()
print(net)

# LOSS AND METRICS (add more methods here if needed)

def bce_loss(y_pred, y_target):
    # TODO: compute binary cross entropy loss from NN output y_pred and target y_target
    bceLoss = nn.BCELoss()
    loss = bceLoss(y_pred, y_target)
    return loss

def binary_acc(y_pred, y_target):
    # TODO: compute accuracy of the NN output y_pred from target y_target
    y_acc = y_pred.clone().detach().requires_grad_(False)
    y_acc[y_pred>0.5] = 1
    y_acc[y_pred<=0.5] = 0
    acc = sum(y_acc==y_target)/len(y_pred)
    return acc

def true_pos(y_pred, y_target):
    y_acc = y_pred.clone().detach().requires_grad_(False)
    y_acc[y_pred>0.5] = 1
    y_acc[y_pred<=0.5] = 0
    truePos = sum(y_acc*y_target)
    return truePos

def true_neg(y_pred, y_target):
    y_acc = y_pred.clone().detach().requires_grad_(False)
    y_acc[y_pred>0.5] = 0
    y_acc[y_pred<=0.5] = 1
    trueNeg = sum(y_acc*(1-y_target))
    return trueNeg

def false_pos(y_pred, y_target):
    y_acc = y_pred.clone().detach().requires_grad_(False)
    y_acc[y_pred>0.5] = 1
    y_acc[y_pred<=0.5] = 0
    falseNeg = sum(y_acc*(1-y_target))
    return falseNeg

def false_neg(y_pred, y_target):
    y_acc = y_pred.clone().detach().requires_grad_(False)
    y_acc[y_pred>0.5] = 0
    y_acc[y_pred<=0.5] = 1
    falseNeg = sum(y_acc*y_target)
    return falseNeg

# TRAINING

# Optimizer definition
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

def train(model, train_loader, optimizer, num_epochs):
    model.train()

    for e in range(1, num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            #(X_batch.shape)
            # TODO: train the model and compute epoch loss and accuracy
            # x.item() returns the number contained within tensor x  
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred_ = model(X_batch)
            y_pred = torch.flatten(y_pred_)

            y_pred = y_pred.to(torch.float32)
            y_batch = y_batch.to(torch.float32)

            loss = bce_loss(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            #break
            epoch_loss += loss
            epoch_acc += acc

        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

train_loader, test_loader = load_data(class_dataset_path)
train(net, train_loader, optimizer, 21)
# VALIDATION


def test(model, test_loader):
    model.eval()

    with torch.no_grad():
        acc = 0
        truePos = 0
        trueNeg = 0
        falsePos = 0
        falseNeg = 0
        corr_pred = 0
        img_1g = torch.zeros((1,3,32,32))

        for X_batch, y_batch in test_loader:
            # TODO: compute the overall accuracy, confusion matrix,
            # precision and recall on the test dataset 
            print(X_batch)
            y_pred_ = model(X_batch)
            y_pred = torch.flatten(y_pred_)

            y_pred = y_pred.to(torch.float32)
            y_batch = y_batch.to(torch.float32)

            truePos += true_pos(y_pred, y_batch)
            trueNeg += true_neg(y_pred, y_batch)
            falseNeg += false_neg(y_pred, y_batch)
            falsePos += false_pos(y_pred, y_batch)
            corr_pred += truePos+trueNeg

            for i in range(len(y_batch)):
                if y_pred[i]>0.5 and y_batch[i]==1:
                    img_1g = X_batch[i,:,:,:]

            #break
        acc = corr_pred/(corr_pred+falseNeg+falsePos)  

        if truePos+falsePos > 0:
            precision = truePos/(truePos+falsePos)  
        else:
            precision = Inf

        if truePos+falseNeg > 0:
            recall = truePos/(truePos+falseNeg)
        else:
            recall = Inf
        print("Metrics in Test Dataset")
        print("Overall accuracy is:", np.array(acc))
        print("True Positives:", np.array(truePos))
        print("True Negatives:", np.array(trueNeg))
        print("False Negatives:", np.array(falseNeg))
        print("False Positives:", np.array(falsePos))
        print("Precision:", np.array(precision))
        print("Recall:", np.array(recall))

        images, labels = one_batch(test_loader)
        plot_images_labels(images, labels)
    return img_1g

# VISUALIZATION

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# obtain one batch of images and labels from data loader
def one_batch(loader):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display
    labels = labels.numpy()
    return images, labels

# display 8 images with labels
def plot_images_labels(images, labels):
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(10, 4))
    
    for idx in np.arange(len(images)):
        ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(labels[idx])
    plt.waitforbuttonpress()


###### 1g 
img = test(net, test_loader)
img = img[None,:]
print(img.shape)
y_pred_ = net(img)
y_pred = torch.flatten(y_pred_)
print(y_pred)
plot_images_labels(img, y_pred)

noise = transforms.ColorJitter(brightness=10, contrast=20, saturation=0, hue=0)
img = noise.forward(img)
y_pred_ = net(img)
y_pred = torch.flatten(y_pred_)
print(y_pred)
plot_images_labels(img, y_pred)
