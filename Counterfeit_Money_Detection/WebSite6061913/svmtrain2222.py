import numpy as np
import cv2
import os
import joblib


target_size = (224, 224)
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def extract_features(image, target_size):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) if len(resized_image.shape) == 3 else resized_image

    gray_image = np.uint8(gray_image)

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(9, 9))
    clahe_image = clahe.apply(gray_image) if np.sum(gray_image) > 0 else gray_image
    denoised_image = cv2.fastNlMeansDenoising(clahe_image, None, h=13, searchWindowSize=16, templateWindowSize=7)

    darkened_image = cv2.convertScaleAbs(denoised_image, alpha=0.7, beta=0)
    region1_coords = (70, 160, 164, 230)
    region2_coords = (10, 200, 118, 133)

    
    region1 = darkened_image[region1_coords[0]:region1_coords[1], region1_coords[2]:region1_coords[3]]
    additional_clahe_1 = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(5, 5))
    preprocessed_region1 = additional_clahe_1.apply(region1) if np.sum(region1) > 0 else region1

    
    region2 = darkened_image[region2_coords[0]:region2_coords[1], region2_coords[2]:region2_coords[3]]
    additional_clahe_2 = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2, 2))
    preprocessed_region2 = additional_clahe_2.apply(region2) if np.sum(region2) > 0 else region2

    watermark_folder = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\watermarkk"
    watermark_images = load_images_from_folder(watermark_folder)

    silverline_folder = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\silverlineeee"
    silverline_images = load_images_from_folder(silverline_folder)

    darkened_image = darkened_image.flatten()

    watermark_threshold = 0.5
    silverline_threshold = 0.5

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

    if is_real:  
        real_count += 1
    else:
        fake_count += 1

    return is_real, preprocessed_image, real_count, fake_count




real_count = 0
fake_count = 0



rimages = load_images_from_folder('C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\train\\real')
for img in rimages:
    result = preprocess_and_save(img, target_size=(224, 224), real_count=real_count, fake_count=fake_count, folder='preprocessed_real')
    is_real, _, real_count, fake_count = result[:4]

fimages = load_images_from_folder('C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\train\\fake')
for img in fimages:
    result = preprocess_and_save(img, target_size=(224, 224), real_count=real_count, fake_count=fake_count, folder='preprocessed_fake')


#
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


class CustomStandardScaler:
    def fit_transform(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        scaled_data = (data - self.mean) / self.std
        return scaled_data

    def transform(self, data):
        return (data - self.mean) / self.std


class CustomSVM:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.weights = np.random.uniform(0, 1, len(X[0]))
        self.bias = np.random.uniform(0, 1)

        
        for epoch in range(self.epochs):
            for features, label in zip(X, y):
                prediction = self.predict(features)
                error = label - prediction

               
                self.weights += self.learning_rate * error * features
                self.bias += self.learning_rate * error

    def predict(self, X):
        self.weights = self.weights.reshape(-1)

        X = X.reshape(-1)

        return 1 if np.dot(self.weights, X) + self.bias > 0 else 0





all_images_real = load_images_from_folder('C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\preprocessed_real')
all_images_fake = load_images_from_folder('C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\preprocessed_fake')
all_images = all_images_real + all_images_fake


all_data = [extract_features(img, target_size) for img in all_images]

all_features = [data[1] for data in all_data]
all_labels = [1 if data[0] else 0 for data in all_data]


train_features, test_features, train_labels, test_labels = custom_train_test_split(
    all_features, all_labels, test_size=0.2, random_state=42
)


scaler = CustomStandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
joblib.dump(scaler, 'scaler_model.joblib')

custom_svm_model = CustomSVM(learning_rate=0.01, epochs=100)
custom_svm_model.fit(train_features, train_labels)


train_predictions = [custom_svm_model.predict(features) for features in train_features]
train_accuracy = sum(pred == label for pred, label in zip(train_predictions, train_labels)) / len(train_labels)
print(f"Accuracy on the training set: {train_accuracy * 100:.2f}%")


test_predictions = [custom_svm_model.predict(features) for features in test_features]
test_accuracy = sum(pred == label for pred, label in zip(test_predictions, test_labels)) / len(test_labels)
print(f"Accuracy on the test set: {test_accuracy * 100:.2f}%")


joblib.dump(custom_svm_model, 'custom_svm_model.joblib')
print("Custom SVM model saved successfully.")

