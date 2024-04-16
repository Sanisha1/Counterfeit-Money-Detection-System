import cv2
import os
import numpy as np

def preprocess_and_crop(image_path, output_folder):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale if it's a multichannel image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Convert image type if necessary
    gray_image = np.uint8(gray_image)

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(9, 9))
    clahe_image = clahe.apply(gray_image) if np.sum(gray_image) > 0 else gray_image

    # Resize image to 224x224
    resized_image = cv2.resize(clahe_image, (224, 224))

    # Specify the coordinates of the two regions you want to crop (adjust as needed)
    region1_coords = (70, 160, 164, 230)
    region2_coords = (10, 200, 118, 133)

    # Extract and save the first region
    region1 = resized_image[region1_coords[0]:region1_coords[1], region1_coords[2]:region1_coords[3]]
    region1_output_path = os.path.join(output_folder, f'region1_{os.path.basename(image_path)}')
    cv2.imwrite(region1_output_path, region1)

    # Extract and save the second region
    region2 = resized_image[region2_coords[0]:region2_coords[1], region2_coords[2]:region2_coords[3]]
    region2_output_path = os.path.join(output_folder, f'region2_{os.path.basename(image_path)}')
    cv2.imwrite(region2_output_path, region2)

# Example usage:
input_folder = 'C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\train'
output_folder = 'C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\alltemplate'

# Iterate through 'real' and 'fake' folders
for label in ['real', 'fake']:
    label_folder = os.path.join(input_folder, label)
    
    # Iterate through images in the label folder
    for filename in os.listdir(label_folder):
        image_path = os.path.join(label_folder, filename)
        preprocess_and_crop(image_path, output_folder)
