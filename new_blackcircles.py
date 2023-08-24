import cv2
import numpy as np

img = cv2.imread('Go2.png')

blur = cv2.GaussianBlur(img, (9,9), 0)

# Convert to grayscale
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)


# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Apply Hough Transform to detect circles
circles = cv2.HoughCircles(eroded, cv2.HOUGH_GRADIENT, dp=1, minDist=7, param1=40, param2=31, minRadius=30, maxRadius=47)

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
cv2.imshow('Lines and Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
                 
