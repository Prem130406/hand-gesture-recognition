import pickle
import numpy as np
import cv2
from flask import Flask, render_template, Response
from hand_gesture_recognition import predict_gesture  # You'll need to implement this function

app = Flask(__name__)

# Load your hand gesture recognition model
model = pickle.load(open("hand_gesture_model.pkl", "rb"))

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform hand gesture recognition on the frame
            gesture = predict_gesture(frame, model)
            
            # Draw the predicted gesture on the frame
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True, port=8000)
