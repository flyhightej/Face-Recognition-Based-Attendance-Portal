from simple_facerec import SimpleFacerec
import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import pandas as pd
from time import time

# Defining Flask App
app = Flask(__name__)

# Number of images to take for each user
nimgs = 1

# Encode faces from a folder
sfr = SimpleFacerec()

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/images'):
    os.makedirs('static/images')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# extract the face from an image
def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(
            gray, 1.2, 5, minSize=(20, 20))
        return face_points
    else:
        return []        

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/images'))

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l   

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}') 
            

################## ROUTING FUNCTIONS #######################
####### for Face Recognition based Attendance System #######

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())


# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    
    # Encode faces from a folder
    sfr.load_encoding_images("static/images/")
    
    # Load Camera
    rtsp_url = 'rtsp://192.0.0.4:8080/h264_ulaw.sdp'
    cap = cv2.VideoCapture(rtsp_url)
    
    
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

       cv2.imshow("Frame", frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())

# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    rtsp_url = 'rtsp://192.0.0.4:8080/h264_ulaw.sdp'
    i, j = 0, 0
    cap = cv2.VideoCapture(rtsp_url)
    start_time = time()
    while time() - start_time < 2:  # Timeout set to 2 seconds
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(newuserid)+'.jpg'
                cv2.imwrite('static/images'+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())   

# Main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)