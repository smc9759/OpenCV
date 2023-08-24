import cv2
import numpy as np

# Load the image
img = cv2.imread('Go2.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Enhance contrast using CLAHE
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)

# Apply Gaussian blur to smooth the image and reduce noise
blurred = cv2.GaussianBlur(enhanced_gray, (7, 7), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Apply Hough Transform to detect circles
# Adjust the parameters to better detect circles in your specific image
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=70, param1=50, param2=30, minRadius=10, maxRadius=50)

# Draw circles
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 2)

# Display the image
cv2.imshow('Detected Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
