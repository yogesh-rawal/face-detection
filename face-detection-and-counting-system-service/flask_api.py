import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

CORS(app, resources={r"/detect_faces": {"origins": "http://localhost:4200"}})

# Paths and configurations
model_dir = "trained_model"  # Directory containing the trained model
UPLOAD_FOLDER = 'uploads'  # Directory to temporarily store uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Allowed file formats
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained Viola-Jones classifier
def load_classifier():
    classifier_file = os.path.join(model_dir, 'cascade.xml')
    if os.path.exists(classifier_file):
        return cv2.CascadeClassifier(classifier_file)
    else:
        raise FileNotFoundError(f"Trained classifier not found: {classifier_file}")

classifier = load_classifier()  # Load the trained classifier once

# Check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Face detection function
def detect_faces(image, classifier):
    """Detect faces in an image using the Viola-Jones algorithm."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Viola-Jones specific parameters
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    print(f"Detected {len(faces)} faces.")
    return image, len(faces)

# Flask route to handle image uploads and face detection
@app.route('/detect_faces', methods=['POST'])
def detect_faces_route():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']

    # Validate file type
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read the uploaded image
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Failed to read the uploaded image'}), 400

        # Detect faces in the image
        result_image, face_count = detect_faces(image, classifier)

        # Save the processed image to return it
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{filename}")
        cv2.imwrite(result_image_path, result_image)

        # Return the processed image as a file download
        return jsonify({
            'processed_image_path': result_image_path,
            'detected_faces_count': face_count
        })

    return jsonify({'error': 'Invalid file type. Only png, jpg, jpeg are allowed.'}), 400

# Flask route to test if API is running
@app.route('/')
def index():
    return "Face detection API is running!"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
