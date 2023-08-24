import cv2 as cv

src = cv.imread('parrot.png')

height, width, channel = src.shape
matrix = cv.getRotationMatrix2D((width/2,height/2),90,1)
dst = cv.warpAffine(src,matrix,(width,height))

cv.imshow("src",src)

cv.imshow("dst", dst)

cv.waitKey(0)

#
