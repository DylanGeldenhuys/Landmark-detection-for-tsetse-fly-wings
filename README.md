# How to





# Tsetse fly wing landmark data for morphometrics (Vol 20,21):
---

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