import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype = 'uint8')

cv.imshow('Blank',blank)

cv.rectangle(blank, (0,0), (blank.shape[1]//2 , blank.shape[0]//2) ,(255,0,0), thickness = cv.FILLED)

cv.imshow('Rectangle', blank)

cv.waitKey(0)
