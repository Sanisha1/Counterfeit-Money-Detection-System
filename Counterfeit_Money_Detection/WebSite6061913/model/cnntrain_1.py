import os
import cv2
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt                                                                                                                           
from keras.models import Model
from keras import regularizers
import tensorflow as tf
from keras.layers import Input,Conv2D, BatchNormalization, Activation, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Dense



def preprocess_input(input_folder, output_folder, watermark_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    watermark_images = [cv2.imread(os.path.join(watermark_folder, file)) for file in os.listdir(watermark_folder) if file.endswith('.jpg') or file.endswith('.png')]

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
                    
                    
                    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

                    equ = cv2.equalizeHist(gray_image)

                    
                    cv2.imwrite(os.path.join(output_class_folder, filename), equ)

                    detect_watermark(equ, watermark_images)



def detect_watermark(equ, watermark_images):
    for template_image in watermark_images:
        
        template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        template_image_gray1 = cv2.equalizeHist(template_image_gray)
        match_result = cv2.matchTemplate(equ, template_image_gray1, cv2.TM_CCOEFF_NORMED)

        
        threshold = 0.7
        max_val = np.max(match_result)
        if max_val > threshold:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
            width, height = template_image_gray1.shape[1], template_image_gray1.shape[0]

            
            #cv2.imshow("Template Image", template_image_gray1)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            
            cv2.rectangle(equ, max_loc, (max_loc[0] + width, max_loc[1] + height), (0, 255, 0), 2)
            print("Watermark Detected!")

            # Display the image with the detected watermark
            #cv2.imshow("Image with Watermark", equ)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            watermark_percentage = round(((max_val+1)/2) * 100, 2)
            print(f"Matching Percentage (Watermark): {watermark_percentage}%")
            break  
        else:
            print('No watermark detected!')



def adjust_contrast_brightness(image, alpha=1.5, beta=3):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def enhance_local_contrast(image, clip_limit=2.0, grid_size=(8, 8)):
    if len(image.shape) == 3 and image.shape[2] in [3, 4]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_image = clahe.apply(image)

    return enhanced_image




def train_model(input_folder, output_folder, watermark_folder):
    preprocess_input(input_folder, output_folder, watermark_folder)

    batch_size = 30

    train_gen = ImageDataGenerator(rescale=1. / 255)
    training_set = train_gen.flow_from_directory(
        directory=output_folder,
        target_size=(310, 665),  
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    class Conv2D:
        def __init__(self, filters, kernel_size, activation=None, input_shape=None, strides=(1, 1), padding='valid'):
            if input_shape:
                self.layer = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, input_shape=input_shape, strides=strides, padding=padding)
            else:
                self.layer = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, strides=strides, padding=padding)
        
        def __call__(self, input_data):
            output_data = self.layer(input_data)
            return output_data

    def MobileNetV2(weights='imagenet', input_shape=(310, 665, 3), num_classes=1000, include_top=False):
        input_tensor = Input(shape=input_shape)

        # Initial Convolution
        x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Building inverted residual blocks
        x = inverted_residual_block(x, 16, (1, 1), t=1, strides=(1, 1))
        x = inverted_residual_block(x, 24, (2, 2), t=6, strides=(2, 2))
        x = inverted_residual_block(x, 32, (3, 3), t=6, strides=(2, 2))
        x = inverted_residual_block(x, 64, (3, 3), t=6, strides=(2, 2))
        x = inverted_residual_block(x, 96, (3, 3), t=6, strides=(1, 1))
        x = inverted_residual_block(x, 160, (3, 3), t=6, strides=(2, 2))
        x = inverted_residual_block(x, 320, (3, 3), t=6, strides=(1, 1))

        # Final convolution
        x = Conv2D(1280, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x =Activation('relu')(x)

        # Global average pooling and dense layer
        x = GlobalAveragePooling2D()(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=input_tensor, outputs=x)

        if include_top:
            x = Dense(num_classes, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=input_tensor, outputs=x)

        if weights is not None:
            
            mobilenet_weights_path='watermark_mobilenetmodel_checkpoint.h5'
            mobilenet_model = MobileNetV2(weights=None, input_shape=(310, 665, 3), include_top=False)
            mobilenet_model.load_weights(mobilenet_weights_path, by_name=True)

        return model

    def inverted_residual_block(input_tensor, filters, kernel_size, t, strides):
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        # Depthwise Convolution
        x = DepthwiseConv2D(kernel_size, strides=strides, padding='same', depth_multiplier=t)(input_tensor)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        # Pointwise Convolution
        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        # Add shortcut if input and output shapes are the same
        if input_tensor.shape[-1] == filters and strides == (1, 1):
            x = tf.keras.layers.add([x, input_tensor])
        return x

    
    mobilenet_model = MobileNetV2(weights='imagenet', input_shape=(310, 665, 3), include_top=False)

    x = mobilenet_model.output
    x = tf.expand_dims(x, axis=-1)
    x = tf.expand_dims(x, axis=-1)
    x = GlobalAveragePooling2D()(x)
    preds = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.001))(x)

    model_final = Model(inputs=mobilenet_model.input, outputs=preds)

    
    model_final = Model(inputs=model_final.input, outputs=preds)

   
    optimizer = SGD(learning_rate=0.001)
    model_final.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

 
    checkpoint = ModelCheckpoint("watermark_mobilenetmodel_checkpoint.h5",
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True,
                                 verbose=1)

   
    hist1 = model_final.fit(
        training_set,
        steps_per_epoch=training_set.samples // batch_size,
        epochs=1,
        callbacks=[checkpoint],  
        workers=10,
        shuffle=True
    )

   
    model_final.save("watermark_mobilenetmodel_final1.h5")
    print("mobilenet_watermark_class_indices", training_set.class_indices)
    f = open("mobilenet_watermark_class_indices.txt", "w")
    f.write(str(training_set.class_indices))
    f.close()

    plt.plot(hist1.history["accuracy"])
    plt.plot(hist1.history['loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "loss"])
    plt.savefig('mobilenet' + '_plot.png')
    plt.show()



train_model(
    input_folder='C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\train',
    output_folder='C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\processed',
    watermark_folder='C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\water',
)



