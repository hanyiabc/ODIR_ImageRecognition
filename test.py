import skimage.io
import skimage.transform
import numpy as np
import cv2

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


img1 = loadAndCropCenterResize('41_left.jpg', (250, 250))
img2 = loadAndCropCenterResizeCV2('41_left.jpg', (250, 250))

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

diff = img1 * 255 - img2

skimage.io.imshow(diff)
skimage.io.show()
skimage.io.imshow(img1)
skimage.io.show()
skimage.io.imshow(img2)
skimage.io.show()