"""Image Recognition App"""
import os
import cv2
import numpy as np
import face_recognition

path = 'images'
data = os.listdir(path)
images = []
names = []
encoded = []

def load_images():
    for i in data:
        images.append(cv2.imread(f'{path}/{i}'))
        names.append(os.path.splitext(i)[0])
 
def get_encodings(images):
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encoded.append(encode)
    return encoded

load_images()
encodings = get_encodings(images)
#Start Web Cam
cap = cv2.VideoCapture(0)
 
while True:
    success, image = cap.read()
    img = cv2.resize(image, (0,0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = face_recognition.face_locations(img)
    face_encoded = face_recognition.face_encodings(img, face)

    for encoded_face,face_location in zip(face_encoded, face):
        matches = face_recognition.compare_faces(encodings, encoded_face)
        distance = face_recognition.face_distance(encodings, encoded_face)
        idx = np.argmin(distance)
 
        if matches[idx]:
            name = names[idx].title()
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(image, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
 
    cv2.imshow('WebCam. Press q to exit', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 