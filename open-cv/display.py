import cv2

# Read an image
img = cv2.imread('example.jpg')  # Replace with your image file name

# Show the image
cv2.imshow('Image', img)

# Wait until a key is pressed
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()

