import cv2

def resize_and_save(input_path, output_path, target_size=(224, 224)):
    # Read the image from the input path
    image = cv2.imread(input_path)

    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)

    # Save the resized image to the output path
    cv2.imwrite(output_path, resized_image)

# Example usage
input_image_path = "C:\\Users\\dell\\Downloads\\Counterfeit_Money_Detection\\WebSite6061913\\train\\real\\train_49.jpg"
output_image_path = "C:\\Users\\dell\\Downloads\\Counterfeit_Money_Detection\\WebSite6061913\\static\\tests\\11real.jpg"
resize_and_save(input_image_path, output_image_path)

