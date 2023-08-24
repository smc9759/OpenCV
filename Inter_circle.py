import cv2
import numpy as np

# Functions for intersection and angle
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

def instance1(img):
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(img, (9,9), 0)

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

    intersections = []
    
    # Draw lines and find intersections
    for i, line1 in enumerate(lines):
        x1, y1, x2, y2 = line1[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        for line2 in lines[i + 1:]:
            pt = intersection(line1[0], line2[0])
            if pt is not None:
                angle_diff = angle(line1[0], line2[0])
                if np.pi / 2 - 0.2 <= angle_diff <= np.pi / 2 + 0.2:  # 90-degree angle (with a 0.2 radian tolerance)
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                    intersections.append((line1[0],pt))
    #angle function = takes two parameters , tuple of coordinates

    return img, intersections

def instance2(img):
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
    return img, black_circles, white_circles

def combine_instances(img, intersections, black_circles, white_circles):
    # Draw lines, intersections, black circles, and white circles on the same image
    img_combined = img.copy()

    # Draw lines and intersections from instance 1
    for line, intersection in intersections:
        x1, y1, x2, y2 = line
        cv2.line(img_combined, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if intersection is not None:
            cv2.circle(img_combined, (int(intersection[0]), int(intersection[1])), 5, (0, 255, 0), -1)

    # Draw black circles from instance 2
    if black_circles is not None:
        for (x, y, r) in black_circles:
            cv2.circle(img_combined, (x, y), r, (0, 255, 0), 2)

    # Draw white circles from instance 2
    if white_circles is not None:
        for (x, y, r) in white_circles:
            cv2.circle(img_combined, (x, y), r, (0, 255, 0), 2)

    return img_combined

# Load the image
img = cv2.imread('Go2.png')

# Call instance1 and instance2 functions and store the results
result_instance1, intersections = instance1(img.copy())
result_instance2, black_circles, white_circles = instance2(img.copy())

# Combine instances into a single image
result_combined = combine_instances(img, intersections, black_circles, white_circles)

# Display the results

cv2.imshow('Combined Result', result_combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
