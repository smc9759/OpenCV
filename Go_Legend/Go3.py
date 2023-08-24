import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('Go2.png')

# Show the original image
cv.imshow('Original Image', img)
cv.waitKey(0)

# Apply Gaussian blur
blur = cv.GaussianBlur(img, (7, 7), 0)  # Increased blur kernel size

# Show the blurred image
cv.imshow('Blurred Image', blur)
cv.waitKey(0)

# Convert the image to grayscale
gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv.Canny(gray, 50, 150)

# Apply Hough Transform to detect lines
lines = cv.HoughLines(edges, 1, np.pi/180, 100)

# Draw lines on the original image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Show the image with detected lines
cv.imshow('Lines', img)
cv.waitKey(0)
cv.destroyAllWindows()
