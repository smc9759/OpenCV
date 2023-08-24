import cv2 as cv

img = cv.imread('Go2.png')

if img is not None and img.size != 0:
    
    cv.imshow('바둑판',img)

    cv.waitKey(0)
else:

    print('Failed to Load Image')
