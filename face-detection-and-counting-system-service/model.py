import cv2
import os
import numpy as np
import random

# Paths
annotations_dir = "FDDB-folds"  # Directory containing FDDB annotation files
data_dir = "originalPics"  # Directory containing original images
categories_dir = "archive/256_ObjectCategories"  # Directory containing category folders
vec_file = "positives.vec"  # File to store vectorized positive samples
bg_file = "bg.txt"  # File containing paths to negative images
model_dir = "trained_model"  # Directory to save the trained model
num_stages = 20  # Number of stages for training the classifier
num_pos = 4000  # Number of positive samples 
num_neg = 3000  # Number of negative samples 

def load_data(annotations_dir, data_dir):
    """Load face annotations from the FDDB dataset."""
    faces = []
    try:
        for fold in range(1, 11):  # Loop through all folds (1-10)
            annotation_file = os.path.join(annotations_dir, f'FDDB-fold-{fold:02d}.txt')
            if not os.path.exists(annotation_file):
                print(f"Annotation file '{annotation_file}' not found.")
                continue
            
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
            
            current_image_path = None
            current_faces = []
            for line in lines:
                stripped_line = line.strip()
                if '/' in stripped_line:  # New image path
                    if current_image_path is not None and current_faces:
                        process_image(current_image_path, current_faces, faces, data_dir)
                    current_image_path = stripped_line
                    current_faces = []  # Reset face data for the new image
                else:  # Face annotations
                    current_faces.append(stripped_line)
            
            # Process the last image in the fold
            if current_image_path is not None and current_faces:
                process_image(current_image_path, current_faces, faces, data_dir)
        
        print(f"Loaded {len(faces)} faces.")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
    
    return faces

def process_image(image_path, face_data, faces, data_dir):
    """Process the image and extract face regions based on annotations."""
    img_full_path = os.path.join(data_dir, image_path + ".jpg")
    img = cv2.imread(img_full_path)
    if img is None:
        print(f"Image not found: {img_full_path}")
        return

    for face in face_data:
        try:
            x, y, a, b, angle = map(float, face.split())
        except ValueError:
            print(f"Error parsing face data: {face}")
            continue
        
        # Calculate bounding box (x, y, w, h)
        x = int(x - a)
        y = int(y - b)
        w = int(2 * a)
        h = int(2 * b)

        # Check if the coordinates are within the image bounds
        if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
            print(f"Skipping invalid bounding box for image {image_path}: ({x}, {y}, {w}, {h})")
            continue

        # Save positive face image in a random category folder
        category = random.choice(os.listdir(categories_dir))
        category_dir = os.path.join(categories_dir, category)
        face_region = img[y:y + h, x:x + w]
        face_file = os.path.join(category_dir, f"{image_path.replace('/', '_')}_{len(faces)}.jpg")
        cv2.imwrite(face_file, face_region)
        faces.append(face_file)

def create_vec_file():
    """Create .vec file for positive samples."""
    # Create a list of all positive samples
    positive_samples = []
    for category in os.listdir(categories_dir):
        category_path = os.path.join(categories_dir, category)
        for img in os.listdir(category_path):
            if img.endswith('.jpg'):
                positive_samples.append(os.path.join(category_path, img))
    
    # Write positive samples to a temporary file
    with open('positives.txt', 'w') as f:
        for sample in positive_samples:
            f.write(f"{sample} 1 0 0 24 24\n")
    
    # Create .vec file
    os.system(f"opencv_createsamples -info positives.txt -num {min(len(positive_samples), num_pos)} -w 24 -h 24 -vec {vec_file}")
    
    # Clean up temporary file
    # os.remove('positives.txt')

def prepare_bg_file():
    """Create background file with paths to negative images."""
    with open(bg_file, 'w') as f:
        for category in os.listdir(categories_dir):
            category_path = os.path.join(categories_dir, category)
            for img in os.listdir(category_path):
                if img.endswith('.jpg'):
                    f.write(f"{os.path.join(category_path, img)}\n")

def train_viola_jones_classifier():
    """Train the Viola-Jones face detector."""
    # Ensure necessary directories exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Prepare the .vec file and background file
    create_vec_file()
    prepare_bg_file()

    # Train the Viola-Jones classifier
    os.system(f"opencv_traincascade -data {model_dir} -vec {vec_file} -bg {bg_file} "
              f"-numPos {num_pos} -numNeg {num_neg} -numStages {num_stages} "
              f"-featureType HAAR -w 24 -h 24 "
              f"-minHitRate 0.995 -maxFalseAlarmRate 0.5")

def load_classifier():
    """Load the trained Viola-Jones classifier."""
    classifier_file = os.path.join(model_dir, 'cascade.xml')
    if os.path.exists(classifier_file):
        return cv2.CascadeClassifier(classifier_file)
    else:
        raise FileNotFoundError(f"Trained classifier not found: {classifier_file}")

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

    print(f"Detected {len(faces)} faces in the image.")
    return faces, image

def main():
    # Load dataset and process face data
    faces = load_data(annotations_dir, data_dir)

    # Train the Viola-Jones classifier
    train_viola_jones_classifier()

    # Load the trained classifier
    classifier = load_classifier()

    # Load an image for detection
    # image_path = "image.png"  # Change this to your test image path
    # image_path="image copy.png"
    # image_path="image copy 2.png"
    # image_path="image copy 3.png"


    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or could not be loaded: {image_path}")

    # Detect faces using Viola-Jones
    detected_faces, result_image = detect_faces(image, classifier)

    # Print total detected faces
    total_detected_faces = len(detected_faces)
    print(f"Total detected faces: {total_detected_faces}")

    # Display the output
    cv2.imshow("Detected Faces (Viola-Jones)", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()