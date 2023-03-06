
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.lib.function_base import flip
import pandas as pd
from math import *
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from skimage.segmentation import disk_level_set


torch.cuda.empty_cache()


class AbstractLandmarksDataset():
    def __init__(self, transform=None,zoom = [1.0 - 0.05, 1.0 + 0.05], rotation = [22], height_shift= [0,0.05], width_shift= [0,0.05], flip_data = True):

        df = pd.read_csv('/Users/dylan.geldenhuys/mlc/tsetse_plos/Landmark-detection-for-tsetse-fly-wings/tsetsedata_2019_left_commas/annotations_left.txt',index_col=0, header=None)

        df2 =  pd.read_csv('/Users/dylan.geldenhuys/mlc/tsetse_plos/Landmark-detection-for-tsetse-fly-wings/tsetsedata_2019_right_commas/annotations_right.txt', index_col= 0, header=None)
        self.tranform = transform
        self.zoom = zoom
        self.rotation = rotation
        self.height_shift = height_shift
        self.width_shift = width_shift
        self.image_filenames = []
        self.landmarks = []
        self.transform = transform
        self.image_dir = '/Users/dylan.geldenhuys/mlc/tsetse_plos/Landmark-detection-for-tsetse-fly-wings/tsetsedata_2019_left_commas/images_left'
        self.image_dir2 = '/Users/dylan.geldenhuys/mlc/tsetse_plos/Landmark-detection-for-tsetse-fly-wings/tsetsedata_2019_right_commas/images_right'
        self.TransF_ = True
        self.flip_data = flip_data

       # ------------------- Append left wings data to dataset class ------------
        
#         print(df)
#         print(df.loc["A001 - 20170120_100715.jpg",1])

       
        for filename in df.index[:]:
            self.image_filenames.append(os.path.join(self.image_dir, filename))
            

            landmarks = []
            for num in range(1, 23, 2):
                
                x_coordinate = df.loc[filename,num] - 1
                y_coordinate = df.loc[filename, num+1] - 1
                landmarks.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmarks)
        

        assert len(self.image_filenames) == len(self.landmarks)

        # ------------------ Append flipped right wings data to dataset class-----

        
        for filename in df2.index[:]:

            self.image_filenames.append(os.path.join(self.image_dir2, filename))

            landmarks = []
            for num in range(1, 23, 2):
                x_coordinate = df2.loc[filename,num] +1
                y_coordinate = df2.loc[filename,num+1] +1
                landmarks.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmarks)
        self.landmarks = np.array(self.landmarks).astype('float32')

        assert len(self.image_filenames) == len(self.landmarks)

        # ----------------------

    def TransF(self):
        self.TransF_ = True
    def NoTransF(self):
        self.TransF_ = False
    def set_params(self, zoom = [0.95, 0.105], rotation = [10], height_shift= [0,0.05], width_shift= [0,0.05]):
        self.zoom = zoom
        self.rotation = rotation
        self.height_shift = height_shift
        self.width_shift = width_shift
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        raise NotImplementedError



class LandmarksDataset(AbstractLandmarksDataset):

    def __init__(self, transform=None,zoom = [1.0 - 0.05, 1.0 + 0.05], rotation = [22], height_shift= [0,0.05], width_shift= [0,0.05], flip_data = True):
        super(LandmarksDataset, self).__init__(transform=transform,zoom = zoom, rotation = rotation, height_shift= height_shift, width_shift= width_shift, flip_data = flip_data)
        

    def __getitem__(self, index):
        params = {'zoom_range': self.zoom, 'rotation_range':self.rotation, 'height_shift_range': self.height_shift, 'width_shift_range': self.width_shift }
        image_ = plt.imread(self.image_filenames[index])
    
        landmarks_ = self.landmarks[index]
        image = plt.imread(self.image_filenames[index])
        
        landmarks = self.landmarks[index]
        
        # Flip image and landmarks "on the fly"
        if self.flip_data and "right" in self.image_filenames[index]:
            image = np.flip(image, axis=1)
            image_ = np.flip(image_, axis=1)
            _,n, _ = image.shape
            landmarks  = np.array([[n - landmarks[i][0], landmarks[i][1]] for i in range(len(landmarks))])
            landmarks_ = np.array([[n - landmarks_[i][0], landmarks_[i][1]] for i in range(len(landmarks_))])
        
        if self.transform and self.TransF_:
            
            image, landmarks = self.transform(image_, landmarks_, params)
            image_shape = image.shape
            landmarks_bool = landmarks < 0
            landmarks_outofbounds = landmarks*224 > image_shape[1] 
            while landmarks_bool.any() or landmarks_outofbounds.any():
                image, landmarks = self.transform(image_, landmarks_, params)
                landmarks_bool = landmarks < 0
                landmarks_outofbounds = landmarks*224 > image_shape[1] 
        else:
            img_shape = image.copy().shape
            image = Image.fromarray(image)
            image = TF.resize(image, (224,224))
            landmarks = torch.tensor(landmarks) / torch.tensor([img_shape[1],img_shape[0]])
            image = TF.to_tensor(image)
            # the following tranform normalises each channel to have a mean at 0.5 and std of 0.5 / NOTE: NOT sure if this is theoreticlly better, should check this
            image = TF.normalize(image, [0.5], [0.5])



        landmarks = torch.tensor(landmarks) - 0.5
        return image, landmarks



class LandmarksMaskDataset(AbstractLandmarksDataset):

    def __init__(self, transform=None,zoom = [1.0 - 0.05, 1.0 + 0.05], rotation = [22], height_shift= [0,0.05], width_shift= [0,0.05], flip_data = True, disk_radius = 2):
        super(LandmarksMaskDataset, self).__init__(transform=transform,zoom = zoom, rotation = rotation, height_shift= height_shift, width_shift= width_shift, flip_data = flip_data)
        self.disk_radius = disk_radius

    def __getitem__(self, index):
        params = {'zoom_range': self.zoom, 'rotation_range':self.rotation, 'height_shift_range': self.height_shift, 'width_shift_range': self.width_shift }
        
        image_ = plt.imread(self.image_filenames[index])
    
        landmarks_ = self.landmarks[index]
        
        image = plt.imread(self.image_filenames[index])
        
        landmarks = self.landmarks[index]
        
        # Flip image and landmarks "on the fly"
        if self.flip_data and "right" in self.image_filenames[index]:
            image = np.flip(image, axis=1)
            image_ = np.flip(image_, axis=1)
            _,n, _ = image.shape
            landmarks  = np.array([[n - landmarks[i][0], landmarks[i][1]] for i in range(len(landmarks))])
            landmarks_ = np.array([[n - landmarks_[i][0], landmarks_[i][1]] for i in range(len(landmarks_))])
        
        if self.transform and self.TransF_:
            
            image, landmarks = self.transform(image_, landmarks_, params)
            image_shape = image.shape
            landmarks_bool = landmarks < 0
            landmarks_outofbounds = landmarks*224 > image_shape[1] 
            while landmarks_bool.any() or landmarks_outofbounds.any():
                image, landmarks = self.transform(image_, landmarks_, params)
                landmarks_bool = landmarks < 0
                landmarks_outofbounds = landmarks*224 > image_shape[1] 
        else:
            img_shape = image.copy().shape
            image = Image.fromarray(image)
            image = TF.resize(image, (224,224))
            landmarks = torch.tensor(landmarks) / torch.tensor([img_shape[1],img_shape[0]])
            image = TF.to_tensor(image)
            # the following tranform normalises each channel to have a mean at 0.5 and std of 0.5 / NOTE: NOT sure if this is theoreticlly better, should check this
            image = TF.normalize(image, [0.5], [0.5])



        
        c, shape, n = image.shape
        landmarks_masks = []
        for landmark in landmarks:
            mask = disk_level_set(image_shape=(shape,shape), center=(landmark[1]*224,landmark[0]*224), radius=self.disk_radius)
            landmarks_masks.append(mask)
        # print(landmarks.shape, image.shape)
        landmarks_masks = np.array(landmarks_masks).astype(np.float32)
        landmarks_masks = torch.from_numpy(landmarks_masks)
        landmarks = torch.tensor(landmarks) - 0.5
        return image, landmarks, landmarks_masks

