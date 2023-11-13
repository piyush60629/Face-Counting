from flask import Flask, render_template, Response
import cv2
import json
import os  # Import the os module
import atexit

app = Flask(__name__)

# Get camera index from environment variable, default to 0
camera_index = int(os.environ.get('CAMERA_INDEX', 0))
cap = cv2.VideoCapture(camera_index)

# Use relative path for the cascade classifier XML file
cascade_path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)

def generate_frames():
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        count = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        yield f"data: {json.dumps({'count': count})}\n\n"

# Explicitly release video capture resources on application exit
def release_resources():
    cap.release()

# Register the release_resources function to be called on application exit
atexit.register(release_resources)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_count')
def update_count():
    return Response(generate_frames(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
