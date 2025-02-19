from simple_facerec import SimpleFacerec
import cv2
import os
from flask import Flask, request, render_template, Response
from datetime import date
from datetime import datetime
import pandas as pd

# Defining Flask App
app = Flask(__name__)

# Encode faces from a folder
sfr = SimpleFacerec()

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}') 


    
def generate_frames():
    
    # Encode faces from a folder
    sfr.load_encoding_images("static/images/")
    
    # Load Camera
    rtsp_url = 'rtsp://192.0.0.4:8080/h264_ulaw.sdp'
    cap = cv2.VideoCapture(rtsp_url)
    print(cap)
    
    
    while True:
       ret, frame = cap.read()

       # Detect Faces
       face_locations, face_names = sfr.detect_known_faces(frame)
       for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            add_attendance(name)
            visible_name = name.split('_')[0]

            cv2.putText(frame, visible_name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
   
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Use a generator to yield the output frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)