import cv2
import numpy as np

# Define the size of the chessboard
chessboard_size = (7, 7)

# Define the locations of the corners on the chessboard
chessboard_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
chessboard_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)

# Load the chessboard image
image = cv2.imread('chessboard.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

# If the corners were found, refine the location of the corners
if ret:
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

    # Draw the corners on the image
    cv2.drawChessboardCorners(image, chessboard_size, corners, ret)

    # Initialize the position of the pieces
    piece_positions = {}

    # Define the colors of the pieces
    piece_colors = {'pawn': (0, 0, 255), 'knight': (0, 255, 0), 'bishop': (255, 0, 0), 'rook': (255, 255, 0), 'queen': (255, 0, 255), 'king': (0, 255, 255)}

    # Iterate over the corners to determine the position of each piece
    for corner in corners:
        x, y = corner[0]
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            color = tuple(image[y, x])
            for piece, piece_color in piece_colors.items():
                if color == piece_color:
                    piece_positions[piece] = (x, y)

    # Print the positions of the pieces
    print(piece_positions)

# Display the image with the corners and piece positions
cv2.imshow('Chessboard', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
