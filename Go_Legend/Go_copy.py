import cv2 as cv
import numpy as np

image = cv.imread('moon.png')

new_image = cv.resize(image,(1200,800))

dst = cv.flip(new_image, 1)

cv.imshow("Moon",dst)

cv.waitKey(0)
