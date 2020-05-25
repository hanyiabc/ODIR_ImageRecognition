import numpy as np
import cv2
import pandas as pd
import skimage.io
import skimage.transform
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

df = pd.read_excel(EXCEL_NAME)
rightImagePaths = df[RIGHT_IMAGE_KEY].to_numpy()
leftImagePaths = df[LEFT_IMAGE_KEY].to_numpy()

def loadAndCropCenterResizeCV2(imgPath, newSize):
    img = cv2.imread(imgPath)
    width, height, ______ = img.shape
    if width == height:
        return cv2.resize(img, newSize)
    length = min(width, height)
    left = (width - length) // 2
    top = (height - length) // 2
    right = (width + length) // 2
    bottom = (height + length) // 2
    return cv2.resize(img[left:right, top:bottom, :], newSize)

def loadAndCropCenterResize(imgPath, newSize):
    img= skimage.io.imread(imgPath)
    width, height, ______ = img.shape
    if width == height:
        return skimage.transform.resize(img, newSize)
    length = min(width, height)
    left = (width - length) // 2
    top = (height - length) // 2
    right = (width + length) // 2
    bottom = (height + length) // 2
    return skimage.transform.resize(img[left:right, top:bottom], newSize)

def cropAndSaveImages(dir, names, saveDir):
    for name in names:
        img = loadAndCropCenterResizeCV2(dir + name, (IMAGE_WIDTH, IMAGE_HEIGHT))
        # skimage.io.imsave(saveDir + name, img, quality=100)
        pngPath = os.path.splitext(saveDir + name)[0] + '.png'
        cv2.imwrite(pngPath, img, )
        # cv2.imwrite(saveDir + name, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
cropAndSaveImages(ORIG_TRAIN_DIRECTORY, rightImagePaths, TRAIN_DIRECTORY)
cropAndSaveImages(ORIG_TRAIN_DIRECTORY, leftImagePaths, TRAIN_DIRECTORY)