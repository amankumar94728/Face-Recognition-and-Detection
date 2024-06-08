import os 
import cv2 as cv
import numpy as np

# Define the directory where images are located
directory = 'photo'

# List of people or categories (e.g., 'cat', 'dog', 'akshay_kumar', 'salman-khan')
people = ['cat', 'dog', 'akshay_kumar', 'salman-khan']

# Load the pre-trained face detection cascade classifier
face_cascade = cv.CascadeClassifier('dec_face.xml')

# Lists to store extracted features (faces) and corresponding labels
features = []
labels = []

def create_train():
    for person in people:
        # Construct the path to the directory containing images for the current person
        path = os.path.join(directory, person)
        
        # Assign a label (category) to the current person
        label = people.index(person)
        
        # Loop through all images in the current person's directory
        for img in os.listdir(path):
            # Check if the file is not a .DS_Store file
            if not img.endswith('.DS_Store'):
                # Construct the full path to the image file
                img_path = os.path.join(path, img) 
                
                # Read the image
                img_array = cv.imread(img_path)
                
                if img_array is not None:
                     # Image loaded successfully, proceed with grayscale conversion
                    gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
                    
                    # Detect faces in the grayscale image
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    
                    # Process each detected face
                    for (x, y, w, h) in faces:
                        # Extract the region of interest (face) from the grayscale image
                        face_roi = gray[y:y+h, x:x+w]
                        
                        # Append the extracted face to the features list
                        features.append(face_roi)
                        
                        # Append the corresponding label to the labels list
                        labels.append(label)
                else:
                    # Handle the case where the image could not be loaded
                    print(f"Error loading image: {img_path}")

# Call the function to create and train the face recognizer
create_train()

# Convert the lists to NumPy arrays for compatibility with OpenCV
features = np.array(features, dtype='object')
labels = np.array(labels)

# Create LBPH Face Recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the face recognizer using the extracted features and labels
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

# Save features and labels as NumPy files
np.save('features.npy', features)
np.save('labels.npy', labels)
