


import numpy as np
import cv2
import os
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

target_size = (224, 224)

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
            print("Watermark detected in Region 1")
            watermark_detected = 1
            break

    if watermark_detected:
        silverline_detected = 0

        for silverline_template in silverline_images:
            silverline_match_region2 = cv2.matchTemplate(preprocessed_region2, silverline_template, cv2.TM_CCOEFF_NORMED)

            if np.max(silverline_match_region2) > silverline_threshold:
                print("Silverline detected in Region 2")
                silverline_detected = 1
                break

        if silverline_detected:
            print("Image is real")
            return True, darkened_image
        else:
            print("Silverline not detected in Region 2")
            print("Image is fake")
            return False, darkened_image
    else:
        print("Watermark not detected in Region 1")
        print("Image is fake")
        return False, darkened_image



def preprocess_and_save(image, target_size, real_count, fake_count, folder):
    is_real, preprocessed_image = extract_features(image, target_size)

    folder = 'preprocessed_fake' if not is_real else 'preprocessed_real'
    index = fake_count if not is_real else real_count

    cv2.imwrite(f'{folder}/{folder}_{index}.png', preprocessed_image.reshape(target_size[0], target_size[1]))
    print(f"Saved {folder} image: {folder}/{folder}_{index}.png")

    if is_real:  # Update the counter only for real images
        real_count += 1
    else:
        fake_count += 1

    return is_real, preprocessed_image, real_count, fake_count



# Initialize counters for saved images
real_count = 0
fake_count = 0

# Load and preprocess real images
rimages = load_images_from_folder('C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\train\\real')
for img in rimages:
    result = preprocess_and_save(img, target_size=(224, 224), real_count=real_count, fake_count=fake_count, folder='preprocessed_real')
    is_real, _, real_count, fake_count = result[:4]

# Load and preprocess fake images
fimages = load_images_from_folder('C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\train\\fake')
for img in fimages:
    result = preprocess_and_save(img, target_size=(224, 224), real_count=real_count, fake_count=fake_count, folder='preprocessed_fake')

# Custom train-test split function

import numpy as np
import cv2
import os
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

target_size = (224, 224)



# Custom train-test split function
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

    return train_features, test_features, train_labels, test_labels


























# Custom standard scaler
class CustomStandardScaler:
    def fit_transform(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        scaled_data = (data - self.mean) / self.std
        return scaled_data

    def transform(self, data):
        # Assuming the scaler has been fit, apply the transformation
        return (data - self.mean) / self.std

# Custom implementation of linear SVM
# Custom implementation of linear SVM
class CustomSVM:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.random.uniform(0, 1, len(X[0]))
        self.bias = np.random.uniform(0, 1)

        # Train the model using stochastic gradient descent
        for epoch in range(self.epochs):
            for features, label in zip(X, y):
                prediction = self.predict(features)
                error = label - prediction

                # Update weights and bias
                self.weights += self.learning_rate * error * features
                self.bias += self.learning_rate * error

    def predict(self, X):

        X = X.reshape(1, -1)

# Perform dot product with reshaped X
        return 1 if np.dot(self.weights, X.T) + self.bias > 0 else 0


# ... (Previous code remains unchanged)

all_images_real = load_images_from_folder('C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\preprocessed_real')
all_images_fake = load_images_from_folder('C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\preprocessed_fake')
all_images = all_images_real + all_images_fake

# Convert all images to arrays
all_data = [extract_features(img, target_size) for img in all_images]

all_features = [data[1] for data in all_data]
all_labels = [1 if data[0] else 0 for data in all_data]

# Custom train-test split with 80-20 split
train_features, test_features, train_labels, test_labels = custom_train_test_split(
    all_features, all_labels, test_size=0.2, random_state=42
)

# Normalize features separately for training and testing sets
scaler = CustomStandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Initialize and train the custom SVM model
custom_svm_model = CustomSVM(learning_rate=0.01, epochs=1000)
custom_svm_model.fit(train_features, train_labels)

# Predictions on the test set
train_predictions = [custom_svm_model.predict(features) for features in train_features]
train_predictions = [1 if pred == 1 else 0 for pred in train_predictions]
train_accuracy = sum(pred == label for pred, label in zip(train_predictions, train_labels)) / len(train_labels)
print(f"Accuracy on the train set: {train_accuracy * 100:.2f}%")
# Predictions on the test set
test_predictions = [custom_svm_model.predict(features) for features in test_features]
test_predictions = [1 if pred == 1 else 0 for pred in test_predictions]
test_accuracy = sum(pred == label for pred, label in zip(test_predictions, test_labels)) / len(test_labels)
print(f"Accuracy on the test set: {test_accuracy * 100:.2f}%")

# Display the number of images in training and testing sets
print(f"Number of images in training set: {len(train_features)}")
print(f"Number of images in testing set: {len(test_features)}")
# Save the model using joblib
joblib.dump(custom_svm_model, 'custom_svm_model.joblib')
print("Custom SVM model saved successfully.")
def calculate_confusion_matrix(predictions, actual_labels):
    true_positive = sum((pred == 1 and actual == 1) for pred, actual in zip(predictions, actual_labels))
    false_positive = sum((pred == 1 and actual == 0) for pred, actual in zip(predictions, actual_labels))
    true_negative = sum((pred == 0 and actual == 0) for pred, actual in zip(predictions, actual_labels))
    false_negative = sum((pred == 0 and actual == 1) for pred, actual in zip(predictions, actual_labels))

    confusion_matrix_result = np.array([[true_negative, false_positive], [false_negative, true_positive]])
    return confusion_matrix_result

def calculate_precision_recall_f(confusion_matrix_result):
    true_positive = confusion_matrix_result[1, 1]
    false_positive = confusion_matrix_result[0, 1]
    false_negative = confusion_matrix_result[1, 0]

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

# Predictions on the test set
test_predictions = [custom_svm_model.predict(features) for features in test_features]

# Calculate the confusion matrix
confusion_matrix_result = calculate_confusion_matrix(test_predictions, test_labels)

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix_result)

# Calculate precision, recall, and f1-score
precision, recall, f1_score = calculate_precision_recall_f(confusion_matrix_result)

# Display precision, recall, and f1-score
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")

# Plot the confusion matrix with annotations
plt.imshow(confusion_matrix_result, cmap='Blues', interpolation='nearest')

# Add annotations for each cell in the matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(confusion_matrix_result[i, j]), ha='center', va='center', color='black', fontsize=12)

plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['Negative', 'Positive'])
plt.show()

# Predict labels on the training set
train_predictions = [custom_svm_model.predict(features) for features in train_features]

# Convert true labels to binary (0 or 1)
train_true_labels = [1 if label == 1 else 0 for label in train_labels]

# Calculate the confusion matrix for training data
conf_matrix_train = calculate_confusion_matrix(train_true_labels, train_predictions)

# Print the confusion matrix for training data
print("Confusion Matrix for Training Data:")
print(conf_matrix_train)
# Calculate precision, recall, and f1-score
precision, recall, f1_score = calculate_precision_recall_f(conf_matrix_train)

# Display precision, recall, and f1-score
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")

# Plot the confusion matrix with annotations
plt.imshow(conf_matrix_train, cmap='Blues', interpolation='nearest')
# Add annotations for each cell in the matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix_train[i, j]), ha='center', va='center', color='black', fontsize=12)

plt.title('Confusion Matrix for Training Data')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['Negative', 'Positive'])
plt.show()


# Plot bar plots for training and testing accuracy
categories = ['Training Accuracy', 'Testing Accuracy']
accuracies = [train_accuracy, test_accuracy]

plt.bar(categories, accuracies, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.ylim(0, 1)  # Set the y-axis limit to be between 0 and 1 for accuracy
plt.show()

from sklearn.metrics import classification_report



train_report = classification_report(train_labels, train_predictions, target_names=['Negative', 'Positive'], output_dict=True)


# Generate classification report for testing set
test_report = classification_report(test_labels, test_predictions, target_names=['Negative', 'Positive'], output_dict=True)


# Display Accuracy for Training Data Table
print("Accuracy for Training Data Table:")
print("-------------------------------------------------------------")
print("|         | Precision |  Recall  | F1-Score |  Support  |")
print("-------------------------------------------------------------")
for label in ['Negative', 'Positive']:
    metrics = train_report[label] 
    print(f"|  {label}   |   {metrics['precision']:.2f}    |   {metrics['recall']:.2f}   |   {metrics['f1-score']:.2f}   |   {metrics['support']}    |")
print("-------------------------------------------------------------")
print(f"| Accuracy|                                 {train_report['accuracy']:.2f}                           |")
print("-------------------------------------------------------------")
macro_avg_metrics = train_report['macro avg']
weighted_avg_metrics = train_report['weighted avg']
print(f"|Macro Avg|   {macro_avg_metrics['precision']:.2f}    |   {macro_avg_metrics['recall']:.2f}   |   {macro_avg_metrics['f1-score']:.2f}   |   {macro_avg_metrics['support']}    |")
print(f"|Weighted Avg|   {weighted_avg_metrics['precision']:.2f}    |   {weighted_avg_metrics['recall']:.2f}   |   {weighted_avg_metrics['f1-score']:.2f}   |   {weighted_avg_metrics['support']}    |")



# Display Accuracy for Testing Data Table
print("\nAccuracy for Testing Data Table:")
print("-------------------------------------------------------------")
print("|         | Precision |  Recall  | F1-Score |  Support  |")
print("-------------------------------------------------------------")
for label in ['Negative', 'Positive']:
    metrics = test_report[label] 
    print(f"|  {label}   |   {metrics['precision']:.2f}    |   {metrics['recall']:.2f}   |   {metrics['f1-score']:.2f}   |   {metrics['support']}    |")
print("-------------------------------------------------------------")
print(f"| Accuracy|                                 {test_report['accuracy']:.2f}                           |")
print("-------------------------------------------------------------")
macro_avg_metrics = test_report['macro avg']
weighted_avg_metrics = test_report['weighted avg']
print(f"|Macro Avg|   {macro_avg_metrics['precision']:.2f}    |   {macro_avg_metrics['recall']:.2f}   |   {macro_avg_metrics['f1-score']:.2f}   |   {macro_avg_metrics['support']}    |")
print(f"|Weighted Avg|   {weighted_avg_metrics['precision']:.2f}    |   {weighted_avg_metrics['recall']:.2f}   |   {weighted_avg_metrics['f1-score']:.2f}   |   {weighted_avg_metrics['support']}    |")






# Save the model using joblib
joblib.dump(custom_svm_model, 'custom_svm_model.joblib')
print("Custom SVM model saved successfully.")
