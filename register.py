import json
import os
import cv2
import numpy as np

with open("./data/details.json") as file:
    json_data = json.load(file)

name = input("Enter the name: ")
img_no = 0
id = len(json_data)

face_cascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer.create()

if os.path.isfile("./data/recognizer.yml"):
    recognizer.read("./data/recognizer.yml")

capture = cv2.VideoCapture(0)
cv2.namedWindow("Face detection", cv2.WINDOW_AUTOSIZE)

while True:
    _, frame = capture.read()
    frame_flip = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame_flip, (x, y), (x+w, y+h), (0, 0, 255), 2)

    text = f"No of pics taken: {img_no}"
    cv2.putText(frame_flip, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Face detection", frame_flip)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if len(detected_faces) != 0:
            recognizer.update([frame_gray[y: y+h, x: x+w]], np.array(id))
            json_data[str(id)] = name
            id += 1
            img_no += 1
    elif key == ord('q'):
        break

recognizer.write("./data/recognizer.yml")
with open("./data/details.json", "w") as file:
    json.dump(json_data, file, indent=4)

print("Registration complete.")

capture.release()
cv2.destroyWindow("Face detection")
