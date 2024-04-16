import cv2
import os

# Load the image
image = cv2.imread('C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\train_processed\\real\\PSI-G-8115.jpg_1.jpg')

# Define the coordinates of the bounding box
x, y, width, height = 115, 25, 17, 178  # right #buttom #left#buttom

# Extract the Region of Interest (ROI) using slicing
roi = image[y:y+height, x:x+width]

# Create a folder to save bounding box images if it doesn't exist
output_folder = 'C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\silver_boxes'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the bounding box as a separate image in the output folder
output_path = os.path.join(output_folder, 'bounding3.jpg')
cv2.imwrite(output_path, roi)

# Perform object detection or analysis within the ROI
# You can use image processing or machine learning techniques here

# Display the original image with the bounding box
cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 1)

# Display the ROI if needed
cv2.imshow('ROI', roi)

# Display the original image with the bounding box and ROI
cv2.imshow('Image with Bounding Box', image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
