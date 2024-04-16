import os
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt                                                                                                                          
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from shutil import copyfile
from sklearn.model_selection import train_test_split
def preprocess_input(input_folder, output_folder, silverline_folder, train_output_folder, val_output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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


                    image = cv2.imread(img_path)

 
                    resized_image = cv2.resize(image, (665, 310))
                    
                    cv2.imwrite(os.path.join(output_class_folder, filename), resized_image)

                    detect_silverline(resized_image, silverline_images)

    for class_name in os.listdir(output_folder):
        class_folder = os.path.join(output_folder, class_name)

        if os.path.isdir(class_folder):
            images = [filename for filename in os.listdir(class_folder) if filename.endswith('.jpg') or filename.endswith('.png')]

            train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

            train_folder = os.path.join(train_output_folder, class_name)
            os.makedirs(train_folder, exist_ok=True)
            for filename in train_images:
                img_path = os.path.join(class_folder, filename)
                copyfile(img_path, os.path.join(train_folder, filename))

            val_folder = os.path.join(val_output_folder, class_name)
            os.makedirs(val_folder, exist_ok=True)
            for filename in val_images:
                img_path = os.path.join(class_folder, filename)
                copyfile(img_path, os.path.join(val_folder, filename))

def detect_silverline(resized_image, silverline_images):
    for template_image in silverline_images:

        match_result = cv2.matchTemplate(resized_image, template_image, cv2.TM_CCOEFF_NORMED)

        threshold = 0.7
        max_val = np.max(match_result)
        if max_val > threshold:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
            width, height = template_image.shape[1], template_image.shape[0]


            #cv2.imshow("Template Image", template_image_gray1)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


            cv2.rectangle(resized_image, max_loc, (max_loc[0] + width, max_loc[1] + height), (0, 255, 0), 2)
            print("Silverline Detected!")

 
            #cv2.imshow("Image with silverline", resized_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


            silverline_percentage = round(max_val * 100, 2)
            print(f"Matching Percentage (Silverline): {silverline_percentage}%")
            break  
        else:
            print('No silverline detected!')
def custom_model(input_shape=(310, 665, 3), num_classes=2):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_custom_model(train_folder, val_folder):
    batch_size = 30

    train_gen = ImageDataGenerator(rescale=1. / 255)
    training_set = train_gen.flow_from_directory(
        directory=train_folder,
        target_size=(310, 665),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    validation_gen = ImageDataGenerator(rescale=1. / 255)
    validation_set = validation_gen.flow_from_directory(
        directory=val_folder,
        target_size=(310, 665),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    model = custom_model()


    optimizer = SGD(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


    checkpoint = ModelCheckpoint("custom_model_class_final.h5",
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True,
                                 verbose=1)
    hist1 = model.fit(
        training_set,
        steps_per_epoch=training_set.samples // batch_size,
        epochs=1,
        validation_data=validation_set,
        validation_steps=validation_set.samples // batch_size,
        callbacks=[checkpoint],
        workers=10,
        shuffle=True
    )


    model.save("custom_model_silver_class_final.h5")
    print("custom_model_silver_class_indices", training_set.class_indices)
    f = open("custom_model_silver_class_indices.txt", "w")
    f.write(str(training_set.class_indices))
    f.close()


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hist1.history["accuracy"], label="Training Accuracy")
    plt.plot(hist1.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist1.history["loss"], label="Training Loss")
    plt.plot(hist1.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
input_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\train'
output_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\resized'
watermark_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\silver'
train_output_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\strain'
val_output_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\svalid'

preprocess_input(input_folder, output_folder, watermark_folder, train_output_folder, val_output_folder)

train_custom_model(train_output_folder, val_output_folder)

