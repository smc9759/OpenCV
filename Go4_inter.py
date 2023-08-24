import cv2
import numpy as np

def intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:  # lines are parallel or coincident
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return px, py

def angle(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    angle1 = np.arctan2(y2 - y1, x2 - x1)
    angle2 = np.arctan2(y4 - y3, x4 - x3)
    angle_diff = np.abs(angle1 - angle2)
    return angle_diff

# Load the image
img = cv2.imread('Go4.png')

# Resize the image
scale_percent = 120  # percentage of the original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Apply Gaussian blur
blur = cv2.GaussianBlur(img, (7,7), 0)

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

# Draw lines and find intersections
for i, line1 in enumerate(lines):
    x1, y1, x2, y2 = line1[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    for line2 in lines[i + 1:]:
        pt = intersection(line1[0], line2[0])
        if pt is not None:
            angle_diff = angle(line1[0], line2[0])
            if np.pi / 2 - 0.02 <= angle_diff <= np.pi / 2 + 0.02:  # 90-degree angle (with a 0.2 radian tolerance)
                cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

# Display the image
cv2.imshow('Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
