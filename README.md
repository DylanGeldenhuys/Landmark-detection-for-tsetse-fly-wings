# Description

This repository contains code used to processes a dataset of tsetse fly wing images and locate 11 landmarks on each image using deep learning models. We provide the training scripts as well as the complete pipeline used to process the images. We also include the resulting data from this process. 

The pipeline is brocken up into two steps we call a two tier process. 
# How to
Tier 1. 
The first tier classifies brocken wing images that are specifically missing landmark 4 and 6 which is the majority of brocken wings. 

The training scripts for the models we experimented with can be found in the folder "Classification"

Tier 2. 
The second tier predicts 11 landmarks on the unbroken wings (we call complete wings). The training scripts for both the models we experimented with can be found in the folders regression and segmentation. The final models can be found in data Dyrad (https://datadryad.org/stash/dataset/doi:10.5061%2Fdryad.qz612jmh1)

The complete pipelines for tier 1 and 2 applied on the full dataset (Vol 20 and 21) can be found in the folder called two_tier_pipeline. 

To apply these models the data first needs to be downloaded from the data dryad. 

NOTE: within each jupyter notebook there may be file paths that need to be changed to the location of the downloaded data from data dyrad.

All other files in the route directory are explained below for your convenience. 
analysis-resnet.ipynb - generates pot for errors and procrustes analyses
analysis-segnet.ipynb - generates pot for errors and procrustes analyses
landmarks_dataset.ipynb - classes for generating and loading the dataset
landmarks_transforms.ipynb - trasnformations class applied during training
model_specs.ipynb - a cript to generate the model specifications
Models.ipynb - Model classes from Pytorch
winglength_predictions.ipynb - This script generates a pot of the linear relationship between the predicted winglength and the measured one. 

Most importantly the final landmark dataset can be found in the data/final_clean.csv. This data contains all the landmarks, that have also been inspected after predictions where made to remove any errors. 

All the data for this code may found in the data dyrad repository here https://datadryad.org/stash/dataset/doi:10.5061%2Fdryad.qz612jmh1
---
# Tsetse fly wing landmark data for morphometrics (Vol 20,21):


This data is intended to be used for morphometric analysis on tsetse fly wings. For more information please consult the journal article linked to this dataset. 

The data consists of two parts. That is the landmark coordinate data created via the automatic landmark detection model developed from the linked journal article, along with various biological lab recordings taken during a lab dissection of the tsetse flies. This data is named morphometric_data.csv.

The second part if the data consists of the image dataset used to train the machine learning models that predicted the landmarks for this data. The code and models can be found in the github code repository https://github.com/DylanGeldenhuys/Landmark-detection-for-tsetse-fly-wings.

## Description of the Data and file structure

All the necessary data for morphometric studies can be found in the morphometric_data.csv. The table fields are fully explained in the methods section in Data Dyrad. Directions for use and analysis can be found in the linked article. 

For your convenience we describe the data fields below

vpn - vpn is the filename, edntified by the volume (v), page (p) and number (n) of the fly. The numbers go up to 20 (20 pairs of wings per a page)

cd - Calender year captured

cm - Month of year captured

cy - Calender year

md - Capture method

g - Genus

s - Sex (1=male, 2= female)

c - Ovarian age category i.e. the number of times a female has ovulated, varying from 0 to 7. 

wlm - wing length in millimeters (measured from landmark 1 to 6).

f - Wing fray, varying from 1-6.

lmkr - Number of missing landmarks for the right wing (Not accurate)

lmkr - Number of missing landmarks for the left wing (Not accurate)

hc - Hatchet cell measurement in millimeters (measured from landmark 11 to 7. This was sometimes measured instead of the wing length if landmark 1 or 6 was missing)

left_good - Classifier prediction for the left wing. 1 = complete wing, 0 = incomplete wing.

right_good - Classifier prediction for the right wing. 1 = complete wing, 0 = incomplete wing

l[x/y][#] = The rest of the columns indicate the pixel location of landmark # , with x and y coordinate.


For those interested in developing landmark detections models for tsetse fly wings, we also provide the training data used in our study to train our models for landmark detection. 

## Sharing/access Information

The landmark data was derived from the model outputs given in the following repository

Code repository: https://github.com/DylanGeldenhuys/Landmark-detection-for-tsetse-fly-wings

data repository: https://datadryad.org/stash/dataset/doi:10.5061%2Fdryad.qz612jmh1