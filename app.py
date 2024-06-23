import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__, template_folder='templates')
app.secret_key = '@VenkyDeexu18'
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time,Periods')

expected_teacher_id = "Armani"
expected_password = "18"

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    if img is not None and len(img) > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if len(gray) > 0: 
            face_points = face_detector.detectMultiScale(gray, 1.3, 5)
            return face_points
        else:
            return []
    else:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    periods = df['Periods']
    l = len(df)
    return names, rolls, times, periods, l

def add_attendance(name, periods):
    username = name.split('_')[0]
    userid = name.split('_')[1]

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{datetime.now().strftime("%H:%M:%S")},{periods}')
    else:
        idx = df.index[df['Roll'] == int(userid)][0]
        if periods < 6:
            df.at[idx, 'Periods'] = periods
        df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

def deletefolder(duser):
    pics = os.listdir(duser)

    for i in pics:
        os.remove(duser + '/' + i)

    os.rmdir(duser)

@app.route('/')
def home():
    return render_template('signup.html')

@app.route('/add', methods=['POST'])
def add():
    if request.method == 'POST':
        new_username = request.form.get('name')
        new_userid = request.form.get('id')

        if new_username and new_userid:
            userimagefolder = 'static/faces/' + new_username + '_' + str(new_userid)
            if not os.path.isdir(userimagefolder):
                os.makedirs(userimagefolder)

            cap = cv2.VideoCapture(0)
            i, j = 0, 0
            while 1:
                _, frame = cap.read()
                faces = extract_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 20), 2, cv2.LINE_AA)
                    if j % 10 == 0:
                        name = new_username + '_' + str(i) + '.jpg'
                        cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                        i += 1
                    j += 1
                if j == 500:
                    break
            cap.release()
            cv2.destroyAllWindows
            train_model()
            return render_template('signin.html')
        else:
            return render_template('signup.html', mess='Please provide both username and user ID')

@app.route('/adminlogin')
def admin_login():
    return render_template('view_attendance.html')

@app.route('/take_attendance', methods=['GET', 'POST'])
def take_attendance():
    if request.method == 'POST':
        student_name = request.form.get('name')
        student_id = request.form.get('student_id')
        userlist, names, rolls, l = getallusers()
        if student_name in names and student_id in rolls:
            return redirect(url_for('start'))
        else:
            flash('Invalid student name or ID. Please try again.', 'error')
            return render_template('signin.html', error_message="Incorrect credentials. Please try again.")
    else:
        return render_template('signin.html')
     
@app.route('/action', methods=['POST'])
def perform_action():
    teacher_id = request.form.get('teacher_id')
    password = request.form.get('password')
    action = request.form.get('action')
    
    if teacher_id == expected_teacher_id and password == expected_password:
        if action == 'view_attendance':
            df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
            attendance_html = df.to_html(index=False)
            return render_template('view.html', attendance_html=attendance_html)
        elif action == 'list_users':
            userlist, names, rolls, l = getallusers()
            return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(),
                                   datetoday2=datetoday2, mess='No Users Found')
        else:
            return render_template('view_attendance.html', error_message="Invalid action.")
    else:
        return render_template('view_attendance.html', error_message="Incorrect credentials. Please try again.")
    
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('signup.html', totalreg=totalreg(), datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return render_template('signup.html', totalreg=totalreg(), datetoday2=datetoday2,
                               mess='Error: Could not open webcam')
    
    df_path = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.exists(df_path) or os.path.getsize(df_path) == 0:
        df = pd.DataFrame(columns=['Name', 'Roll', 'Time', 'Periods'])
    else:
        df = pd.read_csv(df_path)

    frame_count = 0
    max_frames = 100 
    processed_persons = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break

        faces = extract_faces(frame)
        faces_array = np.array(faces)
        if faces_array.any():
            (x, y, w, h) = faces_array[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 20), 2, cv2.LINE_AA)
            # if not df.empty:
            if identified_person not in processed_persons:
                idx = df.index[df['Roll'] == int(identified_person.split('_')[1])]
                if idx.empty:
                    add_attendance(identified_person, 1)
                else:
                    current_periods = df.at[idx[0], 'Periods']
                    if current_periods < 6:
                        add_attendance(identified_person, current_periods + 1)
                    else:
                        print("Periods already at maximum")

        try:
            cv2.imshow('Attendance', frame)
        except cv2.error as e:
            print("CV2 Error:", e)
            break
            
        frame_count += 1
        if frame_count >= max_frames:
            break

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return render_template('success.html')


@app.route('/deleteuser',methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)
    train_model()
    userlist,names,rolls,l = getallusers()    
    return render_template('listusers.html',userlist=userlist,names=names,rolls=rolls,l=l,totalreg=totalreg(),datetoday2=datetoday2, mess='No Users Found') 

if __name__ == '__main__':
    app.run(debug=True)
