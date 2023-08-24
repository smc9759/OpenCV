import cv2
import numpy as np

# Load the image
img = cv2.imread('Go2.png')

# Apply Gaussian blur
blur = cv2.GaussianBlur(img, (9,9), 0)

# Convert to grayscale
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(edges, kernel, iterations=4)
eroded = cv2.erode(dilated, kernel, iterations=4)

# Apply Standard Hough Transform to detect lines
lines = cv2.HoughLines(eroded, 1, np.pi/180, 300)

# Draw lines
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
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the image with detected lines
cv2.imshow('Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
