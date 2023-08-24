import cv2 as cv
import numpy as np


def showimage(image_path):


    if image_path is not None and image_path.size != 0:
    
        cv.imshow('바둑판',image_path)


    else:

        print('Failed to Load Image')

    return img



def resizeimage(image_path, ratio = 0.7):

    new_width = int(image_path.shape[1] * ratio)
    new_height = int(image_path.shape[0] * ratio)
    new_dimension = (new_width, new_height)

    return cv.resize(image_path, new_dimension, interpolation=cv.INTER_AREA)






img = cv.imread('Go2.png')

resized_img = resizeimage(img)

#showimage(resized_img)

blank = np.zeros((500,500,3), dtype = 'uint8')

#showimage(blank)

cv.imshow('Go',resized_img);

cv.imshow('Blank',blank)

#blank[:] = 0,255,0

#cv.imshow('Green',blank)

#blank[200:300, 300:400] = 0,0,255

#cv.imshow('Red Rectangle',blank)

cv.rectangle(blank, (0,0),(250,250), (255,0,0), thickness = 2)

cv.imshow('Rectangle', blank)

cv.waitKey(0)
