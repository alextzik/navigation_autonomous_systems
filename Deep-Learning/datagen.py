from PIL import Image
import numpy as np
import pandas as pd
import os, sys

# TODO: enter the folder paths to raw data and the output dataset. Set the Scene variable 
scene = "Scene02" # ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
rgb_dataset_path = "/Users/AlexandrosTzikas/Desktop/hw3/vkitti_2.0.3_rgb/{scene}/clone/frames/{type}/Camera_0/".format(scene=scene, type="rgb")
bbox_path = "/Users/AlexandrosTzikas/Desktop/hw3/vkitti_2.0.3_textgt/{scene}/clone/{type}.txt".format(scene=scene, type="bbox")
class_dataset_path = "/Users/AlexandrosTzikas/Desktop/hw3/class_dataset/"

bbox = pd.read_csv(bbox_path, delim_whitespace=True)

def bb_intersection_over_union(boxA, boxB): 
    # From: https://github.com/adiprasad/unsup-hard-negative-mining-mscoco 
    # @ "boxA" and "boxB" includes [left, top, right, bottom]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # check if boxes are completed separated. 
    if xB-xA < 0 or yB-yA <0: 
        return 0   

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    denominator = float(boxAArea + boxBArea - interArea)
    iou = 0
    if denominator == 0: iou = 0
    else: iou = interArea / denominator
    
    # return the intersection over union value
    return iou

for i in range(len(os.listdir(rgb_dataset_path))):
    img = Image.open(rgb_dataset_path+"rgb_{i}.jpg".format(i=str(i).zfill(5)))
    Iw, Ih = img.size

    pos_crops = []
    neg_boxes = []
    subbox = bbox[(bbox['frame']==i)&(bbox['cameraID']==0)]

    # Gen starting negatives
    for _ in range(len(subbox)):
        boxA = [np.random.randint(0, Iw-100), np.random.randint(0, Ih-100), 0, 0]
        wA = np.random.randint(50, 100)
        hA = np.random.randint(50, 100)
        boxA[2] = boxA[0]+wA
        boxA[3] = boxA[1]+hA
        neg_boxes.append(boxA)

    for _bbox in subbox.iterrows():
        _bbox = _bbox[1]
        boxB = [_bbox['left'], _bbox['top'], _bbox['right'], _bbox['bottom']]
        for _boxA in neg_boxes:
            if bb_intersection_over_union(boxA, boxB) > 0.1:
                neg_boxes.remove(_boxA)

        wB = np.abs(_bbox['right']-_bbox['left'])
        hB = np.abs(_bbox['top']-_bbox['bottom'])

        if wB>50 and hB>50:
            crop_box = img.crop(boxB)
            pos_crops.append(crop_box)


    neg_crops = []
    for j in range(min(len(pos_crops), len(neg_boxes))):
        crop_box = img.crop(neg_boxes[j])
        neg_crops.append(crop_box)

    # Save files
    for j, crop in enumerate(pos_crops):
        crop.save(class_dataset_path+"pos/{i}_{j}.jpg".format(i=i, j=j))

    for j, crop in enumerate(neg_crops):
        crop.save(class_dataset_path+"neg/{i}_{j}.jpg".format(i=i, j=j))


######## Calculate mean and standard deviation across channels
sum_R = 0
sum_G = 0
sum_B = 0
numOfPixels = 0

for i in range(len(os.listdir(rgb_dataset_path))):
    img = Image.open(rgb_dataset_path+"rgb_{i}.jpg".format(i=str(i).zfill(5)))
    sum_R += sum(np.asarray(list(img.getdata(0)))/255)
    sum_G += sum(np.asarray(list(img.getdata(1)))/255)
    sum_B += sum(np.asarray(list(img.getdata(2)))/255)
    numOfPixels += img.size[0]*img.size[1]

mean_R = sum_R/numOfPixels
mean_G = sum_G/numOfPixels
mean_B = sum_B/numOfPixels

sum_R = 0
sum_G = 0
sum_B = 0

for i in range(len(os.listdir(rgb_dataset_path))):
    img = Image.open(rgb_dataset_path+"rgb_{i}.jpg".format(i=str(i).zfill(5)))
    sum_R += sum((np.asarray(list(img.getdata(0)))/255-mean_R)**2)
    sum_G += sum((np.asarray(list(img.getdata(1)))/255-mean_G)**2)
    sum_B += sum((np.asarray(list(img.getdata(2)))/255-mean_B)**2)

sd_R = np.sqrt(sum_R/numOfPixels)
sd_G = np.sqrt(sum_G/numOfPixels)
sd_B = np.sqrt(sum_B/numOfPixels)

print("Mean of R band is", mean_R)
print("St. Dev of R band is", sd_R)

print("Mean of G band is", mean_G)
print("St. Dev of G band is", sd_G)

print("Mean of B band is", mean_B)
print("St. Dev of B band is", sd_B)