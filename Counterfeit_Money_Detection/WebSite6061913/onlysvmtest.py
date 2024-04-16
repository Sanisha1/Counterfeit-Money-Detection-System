from flask import Flask, render_template, request, jsonify, redirect, url_for, session,flash
from util import base64_to_pil 
import numpy as np
import cv2
import os
import joblib
import cv2
import numpy as np
from keras.models import load_model
import sqlite3
import base64
import secrets
from svmtrain2222 import CustomSVM

app = Flask(__name__)

conn = sqlite3.connect('Counterfeit_Detection_final.db', check_same_thread=False)
cursor = conn.cursor()
app.secret_key = secrets.token_hex(16)

svm_model = joblib.load('custom_svm_model.joblib')

# Create table 
def create_table(cursor):

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS currency_images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_data BLOB,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_result1 (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pred_result TEXT,
            main_image BLOB,
            watermark_similarity FLOAT,
            silverline_similarity FLOAT,
            user_id INTEGER,
            image_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (image_id) REFERENCES currency_images(image_id)
        )
    ''')

# Function to insert a detection result into the database with images
def insert_detection_result(cursor,conn,pred_result, main_image, watermark_similarity, silverline_similarity,user_id,image_id):
    cursor.execute('''
        INSERT INTO detection_result1 (pred_result, main_image, watermark_similarity, silverline_similarity,user_id,image_id)
        VALUES (?, ?, ?, ?,?,?)
    ''', (pred_result, main_image, watermark_similarity, silverline_similarity,user_id,image_id))
    conn.commit()


# Function to insert a upload image into database
def insert_currency_image(cursor,conn,image_data,user_id):
    cursor.execute('''
        INSERT INTO currency_images (image_data,user_id)
        VALUES (?,?)
    ''', (image_data,user_id))
    conn.commit()

    cursor.execute('SELECT last_insert_rowid()')
    image_id = cursor.fetchone()[0]

    return image_id

def save_signup_data(cursor,conn,username,email,password):
    cursor.execute('''
        INSERT INTO users (username, email,password) 
        VALUES (?, ?, ?)
    ''', (username, email, password))

    conn.commit()

def authenticate_user(cursor,email, password):
    cursor.execute('SELECT user_id, password FROM users WHERE email = ? AND password = ?', (email,password))
    result = cursor.fetchone()
    return result
    


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def extract_features(image,threshold, target_size):
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


    watermark_detected = 0

    for watermark_template in watermark_images:
        watermark_match_region1 = cv2.matchTemplate(preprocessed_region1, watermark_template, cv2.TM_CCOEFF_NORMED)

        if np.max(watermark_match_region1) > threshold:
            watermark_detected = 1
            break

    if watermark_detected:
        silverline_detected = 0

        for silverline_template in silverline_images:
            silverline_match_region2 = cv2.matchTemplate(preprocessed_region2, silverline_template, cv2.TM_CCOEFF_NORMED)

            if np.max(silverline_match_region2) > threshold:
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



def predict_image(image_path, threshold,target_size=(224, 224)):

    svm_model = joblib.load('custom_svm_model.joblib')

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Unable to load the test image.")
        return None

    # Pass additional arguments to extract_features
    result = extract_features(image,threshold, target_size=target_size)
    is_real, preprocessed_image = result[0], result[1]

    if preprocessed_image is None:
        print("Error: Unable to extract features from the test image.")
        return None

    preprocessed_image = preprocessed_image.flatten().reshape(1, -1)
    preprocessed_image = svm_model.predict(preprocessed_image)

    # Assign label 1 for real images, label 0 for fake images
    prediction = "real" if is_real else "fake"

    return prediction



def preprocess_image(image_path):
    image = cv2.imread(image_path)

    resized_image = cv2.resize(image, (665, 310))

    normalized_image = resized_image / 255.0

    preprocessed_image = np.expand_dims(normalized_image, axis=0)

    return preprocessed_image, resized_image

def perform_watermark_matching(image_path,threshold):
    watermark_model = load_model('custom_model_class_final.h5')
    preprocessed_image, resized_image = preprocess_image(image_path)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(9, 9))
    clahe_image = clahe.apply(gray_image) if np.sum(gray_image) > 0 else gray_image

    denoised_image = cv2.fastNlMeansDenoising(clahe_image, None, h=13, searchWindowSize=16, templateWindowSize=7)


    darkened_image = cv2.convertScaleAbs(denoised_image, alpha=0.7, beta=0)


    predictions = watermark_model.predict(preprocessed_image)

    predicted_class_index = np.argmax(predictions[0])
    result=predict_image(image_path,threshold)

    if predicted_class_index == 1 and result=='real':
        print("Prediction: Watermark found")

        max_watermark_percentage = 0  
        max_matching_template = None  

        for template_image in watermark_images:
            template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

            clahe1 = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(9, 9))
            clahe_image1 = clahe1.apply(template_image_gray) if np.sum(template_image_gray) > 0 else template_image_gray


            denoised_image1 = cv2.fastNlMeansDenoising(clahe_image1, None, h=13, searchWindowSize=16, templateWindowSize=7)


            darkened_image_template = cv2.convertScaleAbs(denoised_image1, alpha=0.7, beta=0)
            match_result = cv2.matchTemplate(darkened_image, darkened_image_template, cv2.TM_CCOEFF_NORMED)

            max_val = np.max(match_result)
            if max_val > threshold and max_val > max_watermark_percentage:
                watermark_percentage = round(((max_val+1)/2) * 100, 2)
                max_watermark_percentage = watermark_percentage
                max_matching_template = template_image

        if max_matching_template is not None:
            print(f"Maximum Matching Percentage (Watermark): {max_watermark_percentage}%")

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
            height, width = max_matching_template.shape[:2]

            top_left = max_loc
            bottom_right = (top_left[0] + width, top_left[1] + height)

            cv2.rectangle(darkened_image, top_left, bottom_right, (0, 255, 0), 2)

            

            return max_watermark_percentage
        else:
            print('No watermark detected!')
            
    elif predicted_class_index == 0:
        print("Prediction: Watermark not found")
        return 0

def perform_silverline_matching(image_path,threshold):
    silverline_model = load_model('custom_model_silver_class_final.h5')
    preprocessed_image1, resized_image1 = preprocess_image(image_path)

    prediction = silverline_model.predict(preprocessed_image1)
    
    predicted_class_index1 = np.argmax(prediction[0])
    if predicted_class_index1 == 1:
        print("Prediction: Silerline found")

        max_silverline_percentage = 0  
        max_matching_template1 = None  

        for template_image1 in silverline_images:


            match_result1 = cv2.matchTemplate(resized_image1, template_image1, cv2.TM_CCOEFF_NORMED)

        
            max_val1 = np.max(match_result1)
            if max_val1 > threshold and max_val1 > max_silverline_percentage:
                silverline_percentage = round(((max_val1+1)/2) * 100, 2)
                max_silverline_percentage = silverline_percentage
                max_matching_template1 = template_image1

        if max_matching_template1 is not None:
            print(f"Maximum Matching Percentage (Silverline): {max_silverline_percentage}%")

 
            min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(match_result1)
            height, width = max_matching_template1.shape[:2]

            top_left = max_loc1
            bottom_right = (top_left[0] + width, top_left[1] + height)


            cv2.rectangle(resized_image1, top_left, bottom_right, (0, 255, 0), 2)

 
            

            return max_silverline_percentage
        else:
            print('No silverline detected!')
    elif predicted_class_index1 == 0:
        print("Prediction: Silverline not found")
        return 0

def get_user_id_from_session():
    return session.get('user_id')

def get_image_id_from_session():
    return session.get('image_id')



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/signup", methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        try:
            # Connect to the SQLite database (or create it if it doesn't exist)
            conn = sqlite3.connect('counterfeit_detection_final.db')

            # Create a cursor object
            cursor = conn.cursor()

            # Create tables
            create_table(cursor)

            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            repassword = request.form['repassword']
            

            # Ensure password and repassword match
            if password != repassword:
                return render_template('signup.html', error='Passwords do not match.')
            

            save_signup_data(cursor,conn,username, email, password)

            return redirect(url_for('signup_success'))
        finally:
            # Close the connection in the finally block
            conn.commit()
            cursor.close()
            conn.close()
    return render_template('signup.html')

@app.route('/signup_success')
def signup_success():
    flash("Sign up successful! Please log in.")
    return redirect(url_for('login'))

@app.route("/login", methods=['GET','POST'])
def login():
    if request.method == 'POST':
        try:
            # Connect to the SQLite database (or create it if it doesn't exist)
            conn = sqlite3.connect('counterfeit_detection_final.db')

            # Create a cursor object
            cursor = conn.cursor()

            # Create tables
            create_table(cursor)
            email = request.form['email']
            password = request.form['password']

            user = authenticate_user(cursor,email, password)

            if user:
                # Store user ID in session
                session['user_id'] = user[0]
                return redirect(url_for('result'))

            return render_template('login.html', error='Invalid username or password')
        finally:
            # Close the connection in the finally block
            conn.commit()
            cursor.close()
            conn.close()
    return render_template('login.html')



@app.route("/result", methods=['GET'])
def result():
    try:
        # Connect to the SQLite database (or create it if it doesn't exist)
        conn = sqlite3.connect('counterfeit_detection_final.db')

        # Create a cursor object
        cursor = conn.cursor()

        # Create tables
        create_table(cursor)
        
        img_path = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\static\\uploadsimage.png"
                # Read the image file as binary data
        with open(img_path, 'rb') as image_file:
            image_data = image_file.read()

        # Encode the binary image data as base64
        base64_image_data = base64.b64encode(image_data).decode('utf-8')
        user_id = get_user_id_from_session()

        # Store image_id in the session
        session['image_id'] = insert_currency_image(cursor, conn, base64_image_data,user_id)
    finally:
        # Close the connection in the finally block
        conn.commit()
        cursor.close()
        conn.close()
    
    return render_template('afterdemo1.html')
    


from flask import jsonify

@app.route('/predictt', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()

        
        image_data = data.get('image')
        threshold = data.get('threshold')

        img = base64_to_pil(image_data)
    
        img_path = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\static\\uploadsimage.png"
        img.save(img_path)

    
        
        result = predict_image(img_path,threshold)
       

        watermark_percentage = perform_watermark_matching(img_path,threshold)
        silverline_percentage = perform_silverline_matching(img_path,threshold)

        session['prediction_result'] = result
        session['watermark_percentage'] = watermark_percentage
        session['silverline_percentage'] = silverline_percentage
        session['threshold'] = threshold
        
        return jsonify({"result": result, "watermark_percentage": watermark_percentage, "silverline_percentage": silverline_percentage, "threshold":threshold, "redirect": "/final_page"})
        

@app.route('/final_page', methods=['GET'])
def final_page():
    try:
        # Connect to the SQLite database (or create it if it doesn't exist)
        conn = sqlite3.connect('counterfeit_detection_final.db')

        # Create a cursor object
        cursor = conn.cursor()

        # Create tables
        create_table(cursor)

        # Retrieve data from the session
        result = session.get('prediction_result')
        watermark_percentage = session.get('watermark_percentage')
        silverline_percentage = session.get('silverline_percentage')
        threshold = session.get('threshold')
        
        
        
        
        img_path = "C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\static\\uploadsimage.png"
        
       
        main_image_data = base64.b64encode(open(img_path, 'rb').read())
        print(f"Watermark Percentage: {watermark_percentage}")
        print(f"Silverline Percentage: {silverline_percentage}")
        user_id = get_user_id_from_session()
        image_id = get_image_id_from_session()

        # Store image_id in the session
        session['image_id'] = image_id
        insert_detection_result(cursor,conn,result, main_image_data, watermark_percentage, silverline_percentage,user_id,image_id)
    finally:
        # Close the connection in the finally block
        cursor.close()
        conn.close()
    return render_template('Finalpage.html', watermark_percentage=watermark_percentage, silverline_percentage=silverline_percentage)



input_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\train'
output_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913'

watermark_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\water'
watermark_images = [cv2.imread(os.path.join(watermark_folder, file)) for file in os.listdir(watermark_folder) if file.endswith('.jpg') or file.endswith('.png')]

silverline_folder = 'C:\\Users\\dell\\OneDrive\\Desktop\\Counterfeit_Money_Detection_Final\\Counterfeit_Money_Detection\\WebSite6061913\\silver'
silverline_images = [cv2.imread(os.path.join(silverline_folder, file)) for file in os.listdir(silverline_folder) if file.endswith('.jpg') or file.endswith('.png')]

if __name__ == '__main__':
    app.run(port=5002, threaded=False, debug=True, use_reloader=False)