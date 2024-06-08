
import numpy as np 
import cv2 as cv

# Load the pre-trained face detection cascade classifier
face_cascade = cv.CascadeClassifier('dec_face.xml')

# List of people or categories (e.g., 'cat', 'dog', 'akshay_kumar', 'salman-khan')
people = ['cat', 'dog', 'akshay_kumar', 'salman-khan']

# Create LBPH Face Recognizer and load the trained model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Load the image
img = cv.imread('Salman-Khan-1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect faces in the grayscale image
faces_react = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces_react:
    faces_roi = gray[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    
cv.imshow('Detected Face', img)
cv.waitKey(0)


