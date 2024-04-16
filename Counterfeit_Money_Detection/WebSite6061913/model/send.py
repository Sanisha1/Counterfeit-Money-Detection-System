import os
import cv2
import numpy as np
import math
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Model
from keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras import layers
# Function to preprocess input images and check for watermarks and silverlines
def preprocess_input_with_watermark_and_silverline_check(input_folder, output_folder, watermark_folder, silverline_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    watermark_images = [cv2.imread(os.path.join(watermark_folder, file)) for file in os.listdir(watermark_folder) if file.endswith('.jpg') or file.endswith('.png')]
    silverline_images = [cv2.imread(os.path.join(silverline_folder, file)) for file in os.listdir(silverline_folder) if file.endswith('.jpg') or file.endswith('.png')]

    for class_name in os.listdir(input_folder):
        class_folder = os.path.join(input_folder, class_name)

        if os.path.isdir(class_folder):
            output_class_folder = os.path.join(output_folder, class_name)
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)

            for filename in os.listdir(class_folder):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(class_folder, filename)
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    # Apply Gaussian blur to the image
                    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

                    # Use adaptive thresholding to create a binary image
                    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 1)
                    resized_image = resize_image(binary_image)

                    # Save the processed image to the output folder
                    cv2.imwrite(os.path.join(output_class_folder, filename), resized_image)

                    source_image = cv2.imread(os.path.join(output_class_folder, filename))

                    # Watermark detection
                    detect_watermark(source_image, watermark_images)

                    # Silverline detection
                    detect_silverline(source_image, silverline_images)

# Function to detect watermark in an image
def detect_watermark(source_image, watermark_images):
    for template_image in watermark_images:
        match_result = cv2.matchTemplate(source_image, template_image, cv2.TM_CCOEFF_NORMED)

        # Threshold the match to identify the region where the object is exactly in the image
        threshold = 0.5
        max_val = np.max(match_result)
        if max_val > threshold:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
            width, height = template_image.shape[1], template_image.shape[0]
            cv2.rectangle(source_image, max_loc, (max_loc[0] + width, max_loc[1] + height), (0, 255, 0), 2)
            print("Watermark Detected!")

            # Calculate and print the matching percentage for watermark
            watermark_percentage = round(((max_val+1)/2) * 100, 2)
            print(f"Matching Percentage (Watermark): {watermark_percentage}%")
            break  # Break after the first watermark detection
        else:
            print('No watermark detected!')

# Function to detect silverline in an image
def detect_silverline(source_image, silverline_images):
    for template_image_1 in silverline_images:
        # Check if the template image is loaded successfully
        if template_image_1 is not None:
            # Ensure template image dimensions are smaller or equal to the source image
            if template_image_1.shape[0] <= source_image.shape[0] and template_image_1.shape[1] <= source_image.shape[1]:
                match_result1 = cv2.matchTemplate(source_image, template_image_1, cv2.TM_CCOEFF_NORMED)

                # Threshold the match to identify the region where the object is exactly in the image
                threshold1 = 0.5
                max_val = np.max(match_result1)
                if max_val > threshold1:
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result1)
                    width, height = template_image_1.shape[1], template_image_1.shape[0]
                    cv2.rectangle(source_image, max_loc, (max_loc[0] + width, max_loc[1] + height), (0, 255, 0), 2)
                    print("Silverline Detected!")

                    # Calculate and print the matching percentage for silverline
                    silverline_percentage = round(((max_val+1)/2) * 100, 2)
                    print(f"Matching Percentage (Silverline): {silverline_percentage}%")
                    break  # Break after the first silverline detection
                else:
                    print('No silverline detected!')
            else:
                print('Template image dimensions are larger than the source image. Skipping silverline detection!')
        else:
            print('Error loading template image. Skipping silverline detection!')

# Function to adjust contrast and brightness of an image
def adjust_contrast_brightness(image, alpha=1.5, beta=3):
    # Adjust contrast and brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

# Function to enhance local contrast of an image
def enhance_local_contrast(image, clip_limit=2.0, grid_size=(8, 8)):
    if len(image.shape) == 3 and image.shape[2] in [3, 4]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the image is of type uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Apply adaptive histogram equalization for enhancing local contrast
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_image = clahe.apply(image)

    return enhanced_image

# Function to resize an image
def resize_image(image, target_size=(224, 224)):
    # Convert grayscale to RGB
    resized_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    resized_image = cv2.resize(resized_image, target_size)
    return resized_image

# Function to train the model
def train_model(input_folder, output_folder, epochs=2, watermark_folder='', silverline_folder=''):
    preprocess_input_with_watermark_and_silverline_check(input_folder, output_folder, watermark_folder, silverline_folder)

    batch_size = 32

    train_gen = ImageDataGenerator(rescale=1./255)
    training_set = train_gen.flow_from_directory(
        directory=output_folder,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    valid_gen = ImageDataGenerator(rescale=1./255)
    valid_set = valid_gen.flow_from_directory(
        directory='C:\\Users\\dell\\Documents\\Counterfeit_Money_Detection\\WebSite6061913\\valid',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Create MobileNetV2 base model
    mobilenet_model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

    x = mobilenet_model.output
    x = GlobalAveragePooling2D()(x)
    preds = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.001))(x)

    model_final = Model(inputs=mobilenet_model.input, outputs=preds)

    # Change the input layer of the model
    model_final = Model(inputs=model_final.input, outputs=preds)

    training_size = 309
    validation_size = 7
    steps_per_epoch = math.ceil(training_size / batch_size)
    validation_steps = math.ceil(validation_size / batch_size)

    # Compile the model
    optimizer = SGD(learning_rate=0.001)
    model_final.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Use ModelCheckpoint to save the model during training
    checkpoint = ModelCheckpoint("watermark_mobilenetmodel_checkpoint.h5",
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True,
                                 verbose=1)

    # Train the model
    hist1 = model_final.fit(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=valid_set,
        validation_steps=validation_steps,
        callbacks=[checkpoint],  # Add the ModelCheckpoint callback
        workers=10,
        shuffle=True
    )

    # Save the final model after training
    model_final.save("watermark_mobilenetmodel_final.h5")
    print("mobilenet_watermark_class_indices", training_set.class_indices)
    f = open("mobilenet_watermark_class_indices.txt", "w")
    f.write(str(training_set.class_indices))
    f.close()

    plt.plot(hist1.history["accuracy"])
    plt.plot(hist1.history['val_accuracy'])
    plt.plot(hist1.history['loss'])
    plt.plot(hist1.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.savefig('mobilenet' + '_plot.png')
    plt.show()

# Example usage
train_model(
    input_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\train',
    output_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913',
    epochs=2,
    watermark_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\real_bboxes',
silverline_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\silver_boxes'
)