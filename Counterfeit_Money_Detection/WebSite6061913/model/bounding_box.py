import os
import cv2

# Specify the folder path containing the images
input_folder = 'C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\train_processed\\real'

# Specify the folder to save the bounding boxes
output_folder = 'C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\real_bboxes'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image (you may need to adjust the file extensions)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Define the coordinates of the bounding box
        x, y, width, height = 170,67,38,90  #right #buttom #left#buttom
# 170,67,38,83(watermark box)
        # Extract the Region of Interest (ROI) using slicing
        roi = image[y:y + height, x:x + width]

        # Generate a unique name for each bounding box based on the image filename
        output_filename = os.path.splitext(filename)[0] + '_bbox.jpg'
        output_path = os.path.join(output_folder, output_filename)

        # Save the bounding box as a separate image in the output folder
        cv2.imwrite(output_path, roi)

        # Display the bounding box on the original image (optional)
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 1)

        # Display the original image with the bounding box (optional)
        cv2.imshow('Image with Bounding Box', image)

        # Wait for a key press (optional)
        cv2.waitKey(0)

# Close all windows at the end (optional)
cv2.destroyAllWindows()
