import pandas as pd
import skimage.io
import skimage.transform
import numpy as np
import os

EXCEL_NAME = "ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
IMAGE_WIDTH = 384
IMAGE_HEIGHT = 384
IMAGE_CHANNEL = 3
IMAGE_DIMS = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)
LEFT_IMAGE_KEY = 'Left-Fundus'
RIGHT_IMAGE_KEY = 'Right-Fundus'
ORIG_TRAIN_DIRECTORY = 'ODIR-5K_Training_Dataset/'
ORIG_TEST_DIRECTORY = 'ODIR-5K_Testing_Images/'
TRAIN_DIRECTORY = 'train/'
USING_PNG = True

def getMeans(filesLeft, filesRight, directory):
    dataLeft = np.zeros((filesLeft.shape[0], *IMAGE_DIMS))
    dataRight = np.zeros((filesRight.shape[0], *IMAGE_DIMS))
    
    counter = 0
    for idx in range(dataLeft.shape[0]):
        if USING_PNG:
            pathLeft = os.path.splitext(directory + filesLeft[idx])[0] + '.png'
            pathRight = os.path.splitext(directory + filesRight[idx])[0] + '.png'
        else:
            pathLeft = directory + filesLeft[idx]
            pathRight = directory + filesRight[idx]
        dataLeft[counter, :, :, :] = skimage.io.imread(pathLeft)
        dataRight[counter, :, :, :] = skimage.io.imread(pathRight)
    leftMeansR = np.mean(dataLeft[:, :, :, 0][dataLeft[:, :, :, 0] != 0.0])
    leftMeansG = np.mean(dataLeft[:, :, :, 1][dataLeft[:, :, :, 1] != 0.0])
    leftMeansB = np.mean(dataLeft[:, :, :, 2][dataLeft[:, :, :, 2] != 0.0])
    
    rightMeansR = np.mean(dataRight[:, :, :, 0][dataRight[:, :, :, 0] != 0.0])
    rightMeansG = np.mean(dataRight[:, :, :, 1][dataRight[:, :, :, 1] != 0.0])
    rightMeansB = np.mean(dataRight[:, :, :, 2][dataRight[:, :, :, 2] != 0.0])

    return ([leftMeansR, leftMeansG, leftMeansB], [rightMeansR, rightMeansG, rightMeansB])

