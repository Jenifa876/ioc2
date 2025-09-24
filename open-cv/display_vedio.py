import cv2
from flask import Flask, Response

app = Flask(_name_)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Video source
video_source = 'http://192.168.109.180:8080/video'  # Replace with your IP Webcam URL
cap = cv2.VideoCapture(video_source)

def generate_frames():
    while True:
        if not cap.isOpened():
            cap.open(video_source)  # try to reopen stream

        success, frame = cap.read()
        if not success:
            continue  # skip this frame

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "<h1>Face Detection Stream</h1><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if _name_ == "_main_":
    # Use host="0.0.0.0" to access from other devices on the network
    app.run(host="0.0.0.0", port=5004, debug=False)
