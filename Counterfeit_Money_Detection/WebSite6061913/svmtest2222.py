from flask import Flask, render_template, request, jsonify
from util import base64_to_pil 
import numpy as np
import cv2
import os
import joblib
from svmtrain2222 import CustomSVM
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

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

    watermark_folder = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\watermarkk"
    watermark_images = load_images_from_folder(watermark_folder)

    silverline_folder = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\silverlineeee"
    silverline_images = load_images_from_folder(silverline_folder)

    darkened_image = darkened_image.flatten()

    watermark_threshold = 0.7
    silverline_threshold = 0.7

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


app = Flask(__name__)
def predict_image(image_path, target_size=(224, 224)):

    svm_model = joblib.load('custom_svm_model.joblib')

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
    preprocessed_image = svm_model.transform(preprocessed_image)

    # Assign label 1 for real images, label 0 for fake images
    prediction = "real" if is_real else "fake"

    return prediction


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/result", methods=['GET'])
def result():
    return render_template('afterdemo1.html')

@app.route('/predictt', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        img_path = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\static\\uploadsimage.png"
        img.save(img_path)
        prediction = predict_image(img_path)
        result = [prediction]
        return jsonify({"result": result})


if __name__ == '__main__':
    app.run(port=5002, threaded=False, debug=True)

