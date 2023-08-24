import cv2
import numpy as np

# Load the image
img = cv2.imread('Go2.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Enhance contrast using CLAHE
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
#apply CLAHE object to a grayscale image (Mat)
enhanced_gray = clahe.apply(gray)

# Apply Laplacian of Gaussian filter to detect edges
#edges = cv2.GaussianBlur(enhanced_gray, (7, 7), 0)
#edges = cv2.Laplacian(edges, cv2.CV_8U, ksize=3)
#edges = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY)[1]

edges = cv2.Canny(gray, 50, 150)

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Apply Hough Transform to detect circles
circles = cv2.HoughCircles(eroded, cv2.HOUGH_GRADIENT, dp=1, minDist=9, param1=50, param2=30, minRadius=30, maxRadius=50)

# Draw circles and number each circle
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    circle_coords = []  # Initialize a list to store circle coordinates
    for i, (x, y, r) in enumerate(circles):
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        # Draw circle number
        cv2.putText(img, str(i+1), (x-r, y-r), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Save circle center coordinates to the list
        circle_coords.append((x, y))

# Display the image
cv2.imshow('Lines and Circles', eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
