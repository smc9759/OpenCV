import cv2
import numpy as np

#Load Chessboard image
img = cv2.read('chessboard.png')

#Resize the image to be smaller
resize_factor = 0.8
#shape 1 is width shape 2 is RGB
new_width = int(img.shape[1] * resize_factor)
new_height = int(img.shape[0] * resize_factor)
#tuple for width & height
new_dimensions = (new_width , new_height)
img = cv2.resize(img, new_dimensions, interpolation = cv2.INTER_AREA)

# Define board size and number of tiles
board_size = (8,8)
tile_size = img.shape[0] // board_size[0]
#board_size[0] is actually 8. divide the height by 8 and height of one tile
#comes up

#Convert image to grayscale and apply thresholding ( to Bin)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
