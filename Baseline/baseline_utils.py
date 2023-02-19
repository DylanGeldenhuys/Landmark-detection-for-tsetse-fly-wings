
import numpy as np
import torch
import csv
from typing import Tuple
from skimage import io
import matplotlib.pyplot as plt
FIG_SIZE = (6,10)
FIG_SIZE_SUBPLOT = (12,20)
IMG_DIM = (1024, 1280)
IMG_DIM_R = (1280, 0)

def load_annotations(annotations: str, flip_right: bool = False) -> Tuple[list, np.array, int]:
    
    '''
        Reads annotations from disk and flips coordinates if they belong to the
        right-winged image.
        
        Parameters
        --------------
        annotations: str
            file name that hosts the annotations.
            
        flip_right: bool
            boolean indicating whether coordinates should be flipped.
            
        Returns
        --------------
        img_names: list of strings
            file names for the wing images
            
        coordinates: list of floats
            x, y pairs for the 11 landmarks
        
        line_count: int
            number of coordinate sets read in
            
    '''
    
    img_names = []
    coordinates = []
    line_count = 0
    
    # Reading in file
    with open(annotations) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for row in csv_reader:
            img_names.append(row[0])
            float_marks = [float(x) for x in row[1:]]
            coordinates.append(float_marks)
            line_count+=1
         
        print(f'Processed {line_count} lines.')

    assert type(coordinates[0][0]) == float
    coordinates = np.array(coordinates)

    # Flip coordinates for the right wings
    if flip_right:
        img_dim = list(IMG_DIM_R)*11
        img_dim = np.array(img_dim).reshape(-1,22)
        
        # Flip img dimensions. The absolute value is because 
        # IM_DIM_R = (1280,0), and we don't want to subtract 
        # the y values.
        coordinates = abs(img_dim - coordinates)
        
    return img_names, coordinates, line_count



def combine_and_split(left_landmarks: np.array, right_landmarks: np.array) -> Tuple[np.array, np.array]:
    '''
        Accepts left and right landmarks, combines them and creates a seeded random 
        train test and validation split. The seed is set at 42.
        
        
        Parameters
        ----------
        left_landmarks: np.array
            numpy array containing landmarks form left wings
            
        right_landmarks: np.array
            numpy array containing flipped landmarks from right wings
            
            
        Returns
        ---------
        training_landmarks: np.array
            numpy array containing landmarks constituting the training set.
            
            
        test_landmarks: np.array
            numpy array containing landmarks constituting the test set.
    '''
    
    combined_landmarks = np.vstack((left_landmarks, right_landmarks))
    assert combined_landmarks.shape[0] == left_landmarks.shape[0] + right_landmarks.shape[0]

     # Split the data consistent with other methods
    train_dataset, _, test_dataset  = torch.utils.data.random_split(combined_landmarks, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)
    
    
    for item in train_loader:
        training_landmarks = item.numpy()

    for item in test_loader:
        test_landmarks = item.numpy()
        
    return training_landmarks, test_landmarks


def compute_baseline(training_landmarks: np.array, plot_mean: bool = False) -> Tuple[np.array, np.array]:
    '''
        Computes the mean and standard deviation for the x and y
        coordinate values across a given training set.
        
        Parameters:
        --------------
        training_landmarks: np.array
            numpy array constituting training set
            
        plot_mean: bool
            flag to determine whether to plot the mean landmarks
            and error bar for the standard deviation
            
            
        Returns:
        --------------
        mean_landmarks: np.array
            the average x,y values for each landmark
            
        std_landmarks: np.array
            the standard deviation for each landmark
    '''
    
    train_mean = np.mean(training_landmarks, axis=0)
    train_std = np.std(training_landmarks, axis=0)

    assert train_mean.shape == (training_landmarks.shape[1],)
    assert train_std.shape == (training_landmarks.shape[1],)
    
    if plot_mean:
        plt.figure(figsize=FIG_SIZE)
        plt.xlim([0, IMG_DIM[1]])
        plt.ylim([900, 100 ])
        for i in range(0, len(train_mean),2):
            plt.scatter(train_mean[i], train_mean[i+1], c='r')
            plt.errorbar(train_mean[i], train_mean[i+1], xerr=train_std[i], yerr=train_std[i+1], c='b', elinewidth=1)

    
    return train_mean, train_std


def mean_test_error(test_landmarks: np.array, train_mean: np.array) -> np.array:
    '''
        Calculates the mean pixel error over the test set for a 
        given baseline (train_mean).
        
        Parameters
        ------------
        test_landmarks: np.array
            test set containing landmarks
            
        train_mean: np.array
            baseline model using average over training set
            
        Returns
        ------------
        landmark_errors: np.array
            The mean pixel error per predicted coordinate.
    '''
    
    test_landmarks = test_landmarks.reshape(-1,22)
    distance_radicand = (train_mean - test_landmarks)**2
    landmark_errors = np.array([np.sqrt(distance_radicand[:,i] + distance_radicand[:,i+1]) for i in range(0,test_landmarks.shape[1], 2)]).T

    assert landmark_errors.shape == (test_landmarks.shape[0], test_landmarks.shape[1]/2)
    
    return landmark_errors


def sanity_plot(root_path: str,img_names: list, coordinates: np.array, right_img: bool = False) -> None:
    
    '''
        Plot 2 random images and coordinates to check that coordinates
        were read in properly, and flipped appropriately
        
        Parameters
        --------------
        root_path: str
            root for all images
            
        img_names: list of strings
            List of images names
            
        coordinates: np.array
            x,y coordinate pairs
            
        img_right: bool
            boolean to determine whether right image should be flipped
    '''
    
    indices = np.random.randint(len(img_names), size=2)
    _, ax = plt.subplots(1, 2, figsize=FIG_SIZE_SUBPLOT)
    
    for idx,k in zip(indices, range(len(indices))):
        
        image = io.imread(root_path + img_names[idx])
        
        if right_img:
            image = np.flip(image, axis=1)
            
        ax[k].set_title(img_names[idx])
        ax[k].imshow(image)
        
        for i in range(0, len(coordinates[idx]),2):
            ax[k].scatter(coordinates[idx,i], coordinates[idx,i+1], c='y')
    