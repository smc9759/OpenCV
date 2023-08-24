import cv2
import numpy as np

# Load the image
img = cv2.imread('Go2.png')

# Apply Gaussian blur
blur = cv2.GaussianBlur(img, (5,5), 0)

# Convert to grayscale
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Apply Hough Transform to detect lines
lines = cv2.HoughLinesP(eroded, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

# Draw lines on the original image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

# Display the image
cv2.imshow('Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
