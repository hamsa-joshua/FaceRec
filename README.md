# Face Recognition Attendance System

This project is a face recognition-based attendance system built using Python, Flask, and OpenCV. The system allows users to sign up by capturing their face images and then uses these images to recognize them for taking attendance. Administrators can view attendance records, list registered users, and delete user data. 

## Features

- **User Registration**: Users can sign up by entering their name and ID and capturing their face images.
- **Face Recognition**: Recognizes registered users' faces to take attendance.
- **Attendance Management**: Records the time and periods of attendance for each user.
- **Admin Dashboard**: Allows the admin to view attendance records and manage users.

## Requirements

- Python 3.x
- Flask
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/hamsa-joshua/FaceRec.git
    cd FaceRec
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the Haarcascade file:**

    Download the `haarcascade_frontalface_default.xml` file from [OpenCV's GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place it in the root directory of the project.

## Usage

1. **Run the Flask application:**

    ```bash
    python app.py
    ```

2. **Open a web browser and go to `http://127.0.0.1:5000/` to access the application.**

## Directory Structure

```plaintext
FaceRec/
│
├── Attendance/                  # Directory for attendance records
├── static/
│   ├── faces/                   # Directory for storing user face images
│   └── face_recognition_model.pkl  # Trained KNN model for face recognition
├── templates/                   # HTML templates for Flask
│   ├── signup.html              # Signup page
│   ├── signin.html              # Signin page
│   ├── success.html             # Success page after taking attendance
│   ├── view_attendance.html     # Admin login and action selection page
│   ├── view.html                # Page to view attendance records
│   └── listusers.html           # Page to list registered users
├── app.py                       # Main Flask application
├── requirements.txt             # List of required packages
└── README.md                    # Project documentation
```

## Notes

- Ensure that the webcam is connected and accessible.
- The system captures 50 images of each user during registration for better accuracy.
- The attendance records are stored in CSV files under the `Attendance` directory.

Feel free to contribute to this project by opening issues or submitting pull requests on the [GitHub repository](https://github.com/your-username/face-recognition-attendance-system).
