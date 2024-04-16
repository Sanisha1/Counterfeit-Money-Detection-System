import os
import cv2
import numpy as np
import math
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras import layers


def resize_image(image, target_size=(224, 224)):
    resized_image = cv2.resize(image, target_size)
    return resized_image

def preprocess_input_with_watermark_and_silverline_check(input_folder, output_folder, silverline_folder, silverline_threshold):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
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


                    # Silverline detection
                    detect_with_templates(source_image, silverline_folder, "Silverline", silverline_threshold)

def detect_with_templates(source_image, template_folder, detection_type, threshold):
    best_match_score = 0
    best_template = None

    for template_filename in os.listdir(template_folder):
        template_path = os.path.join(template_folder, template_filename)
        template_image = cv2.imread(template_path)

        # Check if the template image is loaded successfully
        if template_image is not None:
            match_result = cv2.matchTemplate(source_image, template_image, cv2.TM_CCOEFF_NORMED)

            # Threshold the match to identify the region where the object is exactly in the image
            max_val = np.max(match_result)
            if max_val > threshold and max_val > best_match_score:
                best_match_score = max_val
                best_template = template_image

    if best_template is not None:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
        width, height = best_template.shape[1], best_template.shape[0]
        cv2.rectangle(source_image, max_loc, (max_loc[0] + width, max_loc[1] + height), (0, 255, 0), 2)
        print(f"{detection_type} Detected using template: {template_filename}")

        # Calculate and print the matching percentage
        matching_percentage = round(((max_val+1)/2) * 100, 2)
        print(f"Matching Percentage ({detection_type}): {matching_percentage}%")
    else:
        print(f'No {detection_type.lower()} detected using any template with threshold: {threshold}.')


# Example usage with separate thresholds

silverline_threshold = 0.5

# Example usage
input_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\train'
output_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913'

silverline_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\silver_boxes'
preprocess_input_with_watermark_and_silverline_check(input_folder, output_folder, silverline_folder,  silverline_threshold)

def preprocess_input_for_generator(image):
    # Apply necessary preprocessing steps
    # (e.g., resizing, watermark, silverline checks)
    processed_image = resize_image(image)
    return processed_image



def train_model(input_folder, output_folder, epochs=2):
    preprocess_input_with_watermark_and_silverline_check(input_folder, output_folder,  silverline_folder,  silverline_threshold)

    batch_size = 32

    # Modify the ImageDataGenerator to use the custom preprocessing function
    train_gen = ImageDataGenerator(preprocessing_function=preprocess_input_for_generator)
    training_set = train_gen.flow_from_directory(
        directory=os.path.join(output_folder, 'train_processed'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input_for_generator)

    valid_set = valid_gen.flow_from_directory(
        directory=os.path.join(output_folder, 'valid'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Create MobileNetV2 base model
    mobilenet_model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    print(mobilenet_model.summary())
    x = mobilenet_model.output

# Equivalent to GlobalAveragePooling2D
    x = layers.GlobalAveragePooling2D()(x)

# Equivalent to Dense(2, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
    x = layers.Dense(2, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(x)

    model_final = Model(inputs=mobilenet_model.input, outputs=x)



    training_size = 65
    validation_size = 7
    steps_per_epoch = math.ceil(training_size / batch_size)
    validation_steps = math.ceil(validation_size / batch_size)

    # Compile the model
    optimizer = SGD(learning_rate=0.001)
    model_final.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Use ModelCheckpoint to save the model during training
    checkpoint = ModelCheckpoint("silverline_mobilenetmodel_checkpoint.h5",
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
    model_final.save("silverline_final.h5")
    print("mobilenet_watermark_class_indices", training_set.class_indices)
    f = open("mobilenet_watermark_class_indices.txt", "w")
    f.write(str(training_set.class_indices))
    f.close()
    
    # Plot training history
    plt.plot(hist1.history["accuracy"])
    plt.plot(hist1.history['val_accuracy'])
    plt.plot(hist1.history['loss'])
    plt.plot(hist1.history['val_loss'])
    plt.title("Model Accuracy and Loss")
    plt.ylabel("Metrics")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "Loss", "Validation Loss"])
    plt.savefig('mobilenet_plot.png')
    plt.show()

train_model(input_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\train',
            output_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913')

