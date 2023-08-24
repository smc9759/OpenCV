import cv2
import numpy as np

# Load the image
img = cv2.imread('Go2.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Enhance contrast using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)

# Apply Laplacian of Gaussian filter to detect edges for black circles
edges = cv2.GaussianBlur(enhanced_gray, (7, 7), 0)
edges = cv2.Laplacian(edges, cv2.CV_8U, ksize=3)
edges = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY)[1]

# Apply morphological operations for black circles
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Apply Hough Transform to detect black circles
black_circles = cv2.HoughCircles(eroded, cv2.HOUGH_GRADIENT, dp=1, minDist=9, param1=50, param2=30, minRadius=30, maxRadius=50)

# Convert to grayscale for white circles
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Enhance contrast using CLAHE for white circles
clahe2 = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
enhanced_gray2 = clahe2.apply(gray2)

# Apply Gaussian blur to smooth the image and reduce noise for white circles
blurred2 = cv2.GaussianBlur(enhanced_gray2, (7, 7), 0)

# Apply Canny edge detection for white circles
edges2 = cv2.Canny(blurred2, 50, 150)

# Apply Hough Transform to detect white circles
white_circles = cv2.HoughCircles(edges2, cv2.HOUGH_GRADIENT, dp=1, minDist=70, param1=50, param2=30, minRadius=10, maxRadius=50)

# Draw circles
if black_circles is not None:
    black_circles = np.round(black_circles[0, :]).astype("int")
    for (x, y, r) in black_circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)

if white_circles is not None:
    white_circles = np.round(white_circles[0, :]).astype("int")
    for (x, y, r) in white_circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        # Removed the line that draws the red dot

# Display the image
cv2.imshow('Lines and Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
