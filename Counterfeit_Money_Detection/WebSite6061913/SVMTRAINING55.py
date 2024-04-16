import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from joblib import dump,load
from svm_model5 import CustomSVM




def preprocess_image(image):
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(9, 9))
    clahe_image = clahe.apply(image) if np.sum(image) > 0 else image
    denoised_image = cv2.fastNlMeansDenoising(clahe_image, None, h=13, searchWindowSize=16, templateWindowSize=7)
    darkened_image = cv2.convertScaleAbs(denoised_image, alpha=0.7, beta=0)
    return darkened_image


def load_images(folder_path, resize_dim=(665, 310)):
    data = []
    template_path = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\water\\b.jpg"
    
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    preprocessed_template = preprocess_image(template)

    for label, class_folder in enumerate(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_folder)
        print(f"Loading images from: {class_path}")

        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)

            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            original_image = cv2.resize(original_image, resize_dim)
            preprocessed_image = preprocess_image(original_image)

            watermark_detected, bounding_box = detect_watermark(preprocessed_image, preprocessed_template)

            label = 1 if watermark_detected else 0

            data.append({'image': preprocessed_image, 'bounding_box': bounding_box, 'label': label})

    labels = [item['label'] for item in data]
    print("Labels:", labels)
    print("Number of unique labels:", len(set(labels)))

    return data


def detect_watermark(image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    threshold = 0.3

    bounding_box = None

    if max_val >= threshold:
        h, w = template.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        bounding_box = (top_left, bottom_right)

    return max_val >= threshold, bounding_box


def load_images1(folder_path, template_folder_path, resize_dim=(665, 310)):
    data1 = []
    templates = []

    for filename in os.listdir(template_folder_path):
        template_path = os.path.join(template_folder_path, filename)
        template = cv2.imread(template_path)
        templates.append(template)

    for label1, class_folder in enumerate(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_folder)
        print(f"Loading images from: {class_path}")

        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)

            original_image = cv2.imread(image_path)
            original_image = cv2.resize(original_image, resize_dim)

            silverline_detected, bounding_box1, max_val1 = detect_silverline(original_image, templates)

            label1 = 1 if silverline_detected else 0

            data1.append({'image1': original_image, 'bounding_box1': bounding_box1, 'label1': label1, 'max_val1': max_val1})

    labels1 = [item1['label1'] for item1 in data1]
    print("Labels:", labels1)
    print("Number of unique labels:", len(set(labels1)))

    return data1


def detect_silverline(image1, templates):
    max_val1 = 0
    bounding_box1 = None
    silverline_detected = False

    for template in templates:
        result1 = cv2.matchTemplate(image1, template, cv2.TM_CCOEFF_NORMED)
        _, temp_max_val, _, temp_max_loc = cv2.minMaxLoc(result1)

        if temp_max_val > max_val1:
            max_val1 = temp_max_val
            top_left = temp_max_loc
            h, w = template.shape[:2]
            bottom_right = (top_left[0] + w, top_left[1] + h)
            bounding_box1 = (top_left, bottom_right)
            silverline_detected = True

    threshold = 0.6

    if max_val1 < threshold:
        silverline_detected = False
        bounding_box1 = None

    return silverline_detected, bounding_box1, max_val1


def custom_train_test_split(features, labels, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)

    data = list(zip(features, labels))
    np.random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]

    train_features, train_labels = zip(*train_data)
    test_features, test_labels = zip(*test_data)

    # Convert tuples to NumPy arrays
    train_features, train_labels = np.array(train_features), np.array(train_labels)
    test_features, test_labels = np.array(test_features), np.array(test_labels)

    return train_features, test_features, train_labels, test_labels

def train_svm(X_train, y_train):
    clf = CustomSVM(learning_rate=0.01, epochs=100)
    clf.fit(X_train, y_train)
    return clf


def main():
    # Path to the folder containing training images
    train_folder_path = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\train"
    # Path to the folder containing watermark templates
    template_folder_path = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\silver"
    # Load images and labels, considering watermark detection
    data = load_images(train_folder_path)

    # Display images with bounding box where watermark is detected
    for item in data:
        image = item['image']
        bounding_box = item['bounding_box']
        
        if bounding_box is not None:
            # Draw bounding box on the image
            cv2.rectangle(image, bounding_box[0], bounding_box[1], 255, 2)

            # Display the image
            #cv2.imshow("Image with Watermark Bounding Box", image)
            #cv2.waitKey(0)
    # Load images and labels, considering watermark detection
    data1 = load_images1(train_folder_path, template_folder_path)

    # Display images with bounding box where watermark is detected
    for item1 in data1:
        image1 = item1['image1']
        bounding_box1 = item1['bounding_box1']
        max_val1 = item1['max_val1']

        if bounding_box1 is not None:
            # Draw bounding box on the image
            cv2.rectangle(image1, bounding_box1[0], bounding_box1[1], 255, 2)

            # Display the image with watermark template match value
            #cv2.imshow(f"Image with Bounding Box (Match Value: {max_val1:.2f})", image1)
            #cv2.waitKey(0)

    #cv2.destroyAllWindows()


    # Combine the results of watermark and silverline detection
    combined_data = []
    for item, item1 in zip(data, data1):
        # Check if both watermark and silverline detections are successful
        if item['label'] == 1 and item1['label1'] == 1:
            label = 1  # Considered as real
        else:
            label = 0  # Considered as fake
            
        combined_data.append(label)

    print("Combined Labels:", combined_data)
    print("Number of Labels 1:", combined_data.count(1))
    print("Number of Labels 0:", combined_data.count(0))

# Split the combined data into feature matrix (X) and target vector (y)
    # Modify the line where the combined feature vector is created
    X_combined = np.array([np.concatenate((item['image'].flatten(), item1['image1'].flatten())) for item, item1 in zip(data, data1)])

    y_combined = np.array(combined_data)  # Use the combined labels

    # Split the combined data into training and testing sets
    X_train_combined, X_test_combined, y_train_combined, y_test_combined = custom_train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42)

    # Train the SVM model
    model_combined = train_svm(X_train_combined, y_train_combined)

    # Save the trained model
    dump(model_combined, 'svmtrain_w_s.joblib')

    # Load the trained model for testing
    model_combined = load('svmtrain_w_s.joblib')


    y_pred_combined = model_combined.predict(X_test_combined)

    # Calculate accuracy
    accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)
    print(f"Accuracy: {accuracy_combined * 100:.2f}%")

if __name__ == "__main__":
    main()

