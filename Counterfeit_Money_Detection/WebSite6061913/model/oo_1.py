from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from util import base64_to_pil 
import os
import joblib
from svmtrain2222 import CustomSVM
from werkzeug.utils import secure_filename
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from flask import Flask, render_template




app = Flask(__name__)
# Load the saved model only once when the script is run
svm_model_path = "C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\custom_svm_model.joblib"
svm_model = joblib.load(svm_model_path)
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images
def resize_image(image, target_size=(224, 224)):
    resized_image = cv2.resize(image, target_size)
    return resized_image
def extract_features(image, target_size):
    # Resize the original image
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) if len(resized_image.shape) == 3 else resized_image

    # Convert image type if necessary
    gray_image = np.uint8(gray_image)

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(9, 9))
    clahe_image = clahe.apply(gray_image) if np.sum(gray_image) > 0 else gray_image
    denoised_image = cv2.fastNlMeansDenoising(clahe_image, None, h=13, searchWindowSize=16, templateWindowSize=7)

    # Darken the image
    darkened_image = cv2.convertScaleAbs(denoised_image, alpha=0.7, beta=0)
    region1_coords = (70, 160, 164, 230)
    region2_coords = (10, 200, 118, 133)

    # Extract and preprocess the first region without resizing
    region1 = darkened_image[region1_coords[0]:region1_coords[1], region1_coords[2]:region1_coords[3]]
    additional_clahe_1 = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(5, 5))
    preprocessed_region1 = additional_clahe_1.apply(region1) if np.sum(region1) > 0 else region1

    # Extract and preprocess the second region without resizing
    region2 = darkened_image[region2_coords[0]:region2_coords[1], region2_coords[2]:region2_coords[3]]
    additional_clahe_2 = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2, 2))
    preprocessed_region2 = additional_clahe_2.apply(region2) if np.sum(region2) > 0 else region2

    watermark_folder = "C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\watermarkk"
    watermark_images = load_images_from_folder(watermark_folder)

    silverline_folder = "C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\silverlineeee"
    silverline_images = load_images_from_folder(silverline_folder)

    darkened_image = darkened_image.flatten()

    watermark_threshold = 0.6
    silverline_threshold = 0.6

    watermark_detected = 0

    for watermark_template in watermark_images:
        watermark_match_region1 = cv2.matchTemplate(preprocessed_region1, watermark_template, cv2.TM_CCOEFF_NORMED)

        if np.max(watermark_match_region1) > watermark_threshold:
            watermark_detected = 1
            break

    if watermark_detected:
        silverline_detected = 0

        for silverline_template in silverline_images:
            silverline_match_region2 = cv2.matchTemplate(preprocessed_region2, silverline_template, cv2.TM_CCOEFF_NORMED)

            if np.max(silverline_match_region2) > silverline_threshold:
                silverline_detected = 1
                break

        if silverline_detected:
            print("Image is real")
            return True, darkened_image
        else:
            print("Image is fake")
            return False, darkened_image
    else:
        print("Image is fake")
        return False, darkened_image



def predict_image(image_path, target_size=(224, 224)):



    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Unable to load the test image.")
        return None

    # Pass additional arguments to extract_features
    result = extract_features(image, target_size=target_size)
    is_real, preprocessed_image = result[0], result[1]

    if preprocessed_image is None:
        print("Error: Unable to extract features from the test image.")
        return None

    preprocessed_image = preprocessed_image.flatten().reshape(1, -1)
    preprocessed_image = svm_model.predict(preprocessed_image)

    # Assign label 1 for real images, label 0 for fake images
    prediction = "real" if is_real else "fake"

    return prediction




@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')
@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')
@app.route("/result", methods=['GET'])
def result():
    return render_template('afterdemo1.html')

@app.route('/predictt', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        img_path = "C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\static\\uploadsimage.png"
        img.save(img_path)
        result = predict_image(img_path)
        return jsonify({"result": result})




def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image to the model input shape
    resized_image = cv2.resize(image, (665, 310))

    # Normalize pixel values to be between 0 and 1
    normalized_image = resized_image / 255.0

    # Expand the dimensions to match the model input shape and include the batch dimension
    preprocessed_image = np.expand_dims(normalized_image, axis=0)

    return preprocessed_image, resized_image

def test_image(model, image_path, watermark_images):
    preprocessed_image, resized_image = preprocess_image(image_path)
    equ_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    equ_image1 = cv2.equalizeHist(equ_image)
        # Make predictions
    predictions = model.predict(preprocessed_image)

    # Get the predicted class index with the highest probability
    predicted_class_index = np.argmax(predictions[0])
        # Display the prediction result
    if predicted_class_index == 1:
        print("Prediction: Watermark found")

        max_watermark_percentage = 0  # Variable to store the maximum watermark percentage
        max_matching_template = None  # Variable to store the template with the maximum watermark percentage

        for template_image in watermark_images:
            template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            template_image_gray = cv2.equalizeHist(template_image_gray)

            match_result = cv2.matchTemplate(equ_image1, template_image_gray, cv2.TM_CCOEFF_NORMED)

        # Threshold the match to identify the region where the object is exactly in the image
            threshold = 0.8
            max_val = np.max(match_result)
            if max_val > threshold and max_val > max_watermark_percentage:
                watermark_percentage = round(max_val * 100, 2)
                max_watermark_percentage = watermark_percentage
                max_matching_template = template_image

        if max_matching_template is not None:
            print(f"Maximum Matching Percentage (Watermark): {max_watermark_percentage}%")

        # Get the location of the detected watermark
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
            height, width = max_matching_template.shape[:2]

            top_left = max_loc
            bottom_right = (top_left[0] + width, top_left[1] + height)

        # Draw a rectangle around the detected watermark
            cv2.rectangle(equ_image1, top_left, bottom_right, (0, 255, 0), 2)

        # Save the image with the detected watermark
            result_image_path = "C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\static\\testimg.jpg"
            cv2.imwrite(result_image_path, equ_image1)

            return max_watermark_percentage
        else:
            print('No watermark detected!')
    elif predicted_class_index == 0:
        print("Prediction: Watermark not found")
        return 0
    
def preprocess_image1(image_path1):
    # Read the image
    image1 = cv2.imread(image_path1)

    # Resize the image to the model input shape
    resized_image1 = cv2.resize(image1, (665, 310))

    # Normalize pixel values to be between 0 and 1
    normalized_image1 = resized_image1 / 255.0

    # Expand the dimensions to match the model input shape and include the batch dimension
    preprocessed_image1 = np.expand_dims(normalized_image1, axis=0)

    return preprocessed_image1, resized_image1

def test_image1(model1, image_path1, silverline_images):
    preprocessed_image1, resized_image1 = preprocess_image(image_path1)

        # Make predictions
    predictions1 = model1.predict(preprocessed_image1)

    # Get the predicted class index with the highest probability
    predicted_class_index1 = np.argmax(predictions1[0])
        # Display the prediction result
    if predicted_class_index1 == 1:
        print("Prediction: Watermark found")

        max_silverline_percentage = 0  # Variable to store the maximum watermark percentage
        max_matching_template1 = None  # Variable to store the template with the maximum watermark percentage

        for template_image1 in silverline_images:


            match_result1 = cv2.matchTemplate(resized_image1, template_image1, cv2.TM_CCOEFF_NORMED)

        # Threshold the match to identify the region where the object is exactly in the image
            threshold1 = 0.8
            max_val1 = np.max(match_result1)
            if max_val1 > threshold1 and max_val1 > max_silverline_percentage:
                silverline_percentage = round(max_val1 * 100, 2)
                max_silverline_percentage = silverline_percentage
                max_matching_template1 = template_image1

        if max_matching_template1 is not None:
            print(f"Maximum Matching Percentage (Watermark): {max_silverline_percentage}%")

        # Get the location of the detected watermark
            min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(match_result1)
            height, width = max_matching_template1.shape[:2]

            top_left = max_loc1
            bottom_right = (top_left[0] + width, top_left[1] + height)

        # Draw a rectangle around the detected watermark
            cv2.rectangle(resized_image1, top_left, bottom_right, (0, 255, 0), 2)

        # Save the image with the detected watermark
            result_image_path1 = "C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\static\\testsilverimg.jpg"
            cv2.imwrite(result_image_path1, resized_image1)

            return max_silverline_percentage
        else:
            print('No watermark detected!')
    elif predicted_class_index1 == 0:
        print("Prediction: Watermark not found")
        return 0
# Load the saved model
saved_model_path = "watermark_mobilenetmodel_final.h5"
loaded_model = load_model(saved_model_path)
watermark_folder = 'C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\water'
watermark_images = [cv2.imread(os.path.join(watermark_folder, file)) for file in os.listdir(watermark_folder) if file.endswith('.jpg') or file.endswith('.png')]

saved_model_path1 = "silverline_mobilenetmodel_final.h5"
loaded_model1 = load_model(saved_model_path1)
silverline_folder = 'C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\silver'
silverline_images = [cv2.imread(os.path.join(silverline_folder, file)) for file in os.listdir(silverline_folder) if file.endswith('.jpg') or file.endswith('.png')]
# Usage
@app.route('/fake_page', methods=['GET'])
def fake_page():
    image_path = "C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\static\\uploadsimage.png"
    image_path1="C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\static\\uploadsimage.png"
    watermark_percentage = test_image(loaded_model, image_path, watermark_images)
    silverline_percentage = test_image1(loaded_model1,image_path1,silverline_images)
    return render_template('Fake-1.html', watermark_percentage=watermark_percentage,silverline_percentage=silverline_percentage)


if __name__ == '__main__':
    app.run(port=5002, threaded=False, debug=True)



