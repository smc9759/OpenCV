import cv2
import numpy as np

def get_pawn_compactness(tile_coords, contours, tile_size):
    pawn_x, pawn_y = tile_coords
    pawn_x_center, pawn_y_center = pawn_x * tile_size + tile_size // 2, pawn_y * tile_size + tile_size // 2

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_center = np.array([x + w // 2, y + h // 2])

        if np.linalg.norm(contour_center - np.array([pawn_x_center, pawn_y_center])) < tile_size * 0.5:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            return compactness

    return None

# Load chessboard image
img = cv2.imread('chessboard.png')

# Resize the image to be smaller (e.g., 50% smaller)
resize_factor = 0.8
new_width = int(img.shape[1] * resize_factor)
new_height = int(img.shape[0] * resize_factor)
new_dimensions = (new_width, new_height)
img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)

# Define board size and number of tiles
board_size = (8, 8)
tile_size = img.shape[0] // board_size[0]

# Convert image to grayscale and apply thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

# Find contours in the thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate compactness thresholds based on white pawn at 0,6
white_pawn_coords = (0, 6)
#img.shape[0] - tile_size * 


white_pawn_compactness = get_pawn_compactness(white_pawn_coords, contours, tile_size)
print(white_pawn_compactness)
if white_pawn_compactness is not None:
    lower_threshold = white_pawn_compactness * 0.8
    upper_threshold = white_pawn_compactness * 1.2
else:
    lower_threshold, upper_threshold = 0.3, 0.6

# Initialize tile number and coordinate arrays
tile_nums = np.zeros((board_size[0], board_size[1]), dtype=np.int32)
coords = np.zeros((board_size[0], board_size[1], 2), dtype=np.int32)

# Loop through each contour and classify as pawn or not
for contour in contours:
    # Calculate contour area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Check if area or perimeter is too small
    if area < 10 or perimeter < 10:
        # Skip contour if area or perimeter is too small
        continue

    # Calculate compactness (circularity) of contour
    compactness = (4 * np.pi * area) / (perimeter ** 2)

    # Classify contour as pawn if compactness is within the calculated range
    if lower_threshold <= compactness <= upper_threshold:
        # Draw bounding rectangle around pawn and highlight in blue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate tile number and save to array
        tx, ty = x // tile_size, y // tile_size
        tile_nums[board_size[0] - 1 - ty, tx] += 1
        coords[board_size[0] - 1 - ty, tx] = [x + w // 2, y + h // 2]

        # Add tile number to center of tile
        text = f'({tx}, {board_size[0] - 1 - ty})'
        cv2.putText(img, text, (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Print tile number array and coordinate array
print(tile_nums)
print(coords)

# Show modified image
cv2.imshow('Chessboard', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
