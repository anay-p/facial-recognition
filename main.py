import json
import cv2

face_cascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("./data/recognizer.yml")

with open("./data/details.json") as file:
    name_data = json.load(file)

capture = cv2.VideoCapture(0)
cv2.namedWindow("Face recognition", cv2.WINDOW_AUTOSIZE)

def change_brightness(new_brightness_value):
    capture.set(10, new_brightness_value)

brightness = capture.get(10)
cv2.createTrackbar("Brightness", "Face recognition", int(brightness), 256, change_brightness)

while cv2.waitKey(1) & 0xFF != ord('q'):
    _, frame = capture.read()
    frame_flip = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(frame_gray[y:y+h, x:x+w])
        if confidence < 50:
            text = f"{name_data[str(id)]} - {round(100-confidence, 2)}%"
            color = (0, 255, 0)
        else:
            text = "Unkown"
            color = (0, 0, 255)
        cv2.putText(frame_flip, text, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        cv2.rectangle(frame_flip, (x, y), (x+w, y+h), color, 2)

    text = f"No of detected faces: {len(faces)}"
    cv2.putText(frame_flip, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Face recognition", frame_flip)

capture.release()
cv2.destroyWindow("Face recognition")
