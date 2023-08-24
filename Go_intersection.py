import cv2
import numpy as np
from sklearn.cluster import DBSCAN

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
img = cv2.imread('Go2.png')

# Apply Gaussian blur
blur = cv2.GaussianBlur(img, (9,9), 0)

# Convert to grayscale
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(edges, kernel, iterations=3)
eroded = cv2.erode(dilated, kernel, iterations=3)


# Apply Hough Transform to detect lines
lines = cv2.HoughLinesP(eroded, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

# Group lines using DBSCAN clustering
coords = []
for line in lines:
    coords.append(line[0])
coords = np.array(coords)
dbscan = DBSCAN(eps=20, min_samples=2, metric='manhattan')
labels = dbscan.fit_predict(coords)

# Draw lines for each cluster
unique_labels = np.unique(labels)
for label in unique_labels:
    cluster_coords = coords[labels == label]
    if cluster_coords.shape[0] < 2:
        continue
    x1, y1, x2, y2 = np.min(cluster_coords[:,0]), np.min(cluster_coords[:,1]), np.max(cluster_coords[:,2]), np.max(cluster_coords[:,3])
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the image
cv2.imshow('Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
