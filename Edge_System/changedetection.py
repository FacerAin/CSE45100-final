import os
import cv2
import pathlib
import requests
from datetime import datetime
import time
import uuid

class ChangeDetection:
    # HOST = 'https://syw5141.pythonanywhere.com'
    # username = 'syw5141'
    # password = '1234'
    # token = 'b8f529195320e4886b7c00777630759ecfa21d7a'
    HOST = 'http://127.0.0.1:8000'
    username = 'syw5141'
    password = '1234'
    token = 'b9506731a52a9285af72fe2de435c41dd891d44a'


    
    def __init__(self, names):
        # Session tracking
        self.session_start_time = datetime.now()
        self.session_id = str(uuid.uuid4())
        
        # Initialize previous detection results
        self.result_prev = [0 for _ in range(len(names))]
        
        # Initialize time trackers
        self.absence_start_time = None
        self.eyes_closed_start_time = None
        self.last_state = None
        
        # State constants
        self.ABSENCE_THRESHOLD = 10  # 10 seconds threshold for absence
        self.EYES_CLOSED_THRESHOLD = 5  # 5 seconds threshold for eyes closed
        
        # Get API token
        res = requests.post(
            self.HOST + '/api-token-auth/',
            {'username': self.username, 'password': self.password}
        )
        res.raise_for_status()
        self.token = res.json()['token']

    def __del__(self):
        """Send session data to server when object is destroyed"""
        try:
            session_end_time = datetime.now()
            
            headers = {'Authorization': 'TOKEN ' + self.token, 'Accept': 'application/json'}
            
            session_data = {
                'session_id': self.session_id,
                'session_start_date': self.session_start_time.isoformat(),
                'session_end_date': session_end_time.isoformat(),
                'author': "1"  # Assuming author ID is 1
            }
            
            # Send session data to server
            res = requests.post(
                self.HOST + '/api_root/Session/',
                data=session_data,
                headers=headers
            )
            print(f"Session data sent. Response: {res.status_code}")
            
        except Exception as e:
            print(f"Error sending session data: {e}")

    def check_state(self, names, detected_current):
        """
        Determine the current state based on detections
        Returns: (state_name, should_send_alert)
        """
        current_time = time.time()
        person_detected = False
        eyes_open = False
        
        # Check detections
        for i, name in enumerate(names.values()):
            if (name.lower() == 'awake' or name.lower() == 'drowsy') and detected_current[i] == 1:
                person_detected = True
            if name.lower() == 'awake' and detected_current[i] == 1:
                eyes_open = True

        # State 1: Person absent for more than 10 seconds
        if not person_detected:
            if self.absence_start_time is None:
                self.absence_start_time = current_time
            if current_time - self.absence_start_time >= self.ABSENCE_THRESHOLD:
                return "LONG_ABSENCE"
        else:
            self.absence_start_time = None

        # State 2: Eyes closed for more than 5 seconds
        if person_detected and not eyes_open:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = current_time
            if current_time - self.eyes_closed_start_time >= self.EYES_CLOSED_THRESHOLD:
                return "EYES_CLOSED_LONG"
        else:
            self.eyes_closed_start_time = None

        # State 3: Person present and eyes open
        return "NORMAL"

    def add(self, names, detected_current, save_dir, image):
        state = self.check_state(names, detected_current)
        
        if self.last_state != state:
            self.last_state = state
            title = f"Student State: {state}"
            text = state
            print(f"Sending alert: {title} - {text}")
            self.send(save_dir, image, title, text)
            
        self.result_prev = detected_current[:]

    def get_state_description(self, state):
        descriptions = {
            "LONG_ABSENCE": "Student has been absent for more than 10 seconds",
            "EYES_CLOSED_LONG": "Student's eyes have been closed for more than 5 seconds",
            "NORMAL": "Student has returned to normal state - present and alert"
        }
        return descriptions.get(state, "Unknown state")

    def send(self, save_dir, image, title, text):
        now = datetime.now()
        today = datetime.now()
        save_path = pathlib.Path(
            os.getcwd()
        ) / save_dir / 'detected' / str(today.year) / str(today.month) / str(today.day)
        save_path.mkdir(parents=True, exist_ok=True)

        full_path = save_path / '{0}-{1}-{2}-{3}.jpg'.format(
            today.hour, today.minute, today.second, today.microsecond
        )

        dst = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(full_path), dst)

        headers = {'Authorization': 'TOKEN ' + self.token, 'Accept': 'application/json'}

        data = {
            'session_id': self.session_id,  # Using session_id as uuid
            'title': title,
            'text': text,
            'published_date': now.isoformat(),
            'author': "1"
        }
        files = {'image': open(full_path, 'rb')}

        try:
            res = requests.post(self.HOST + '/api_root/Post/', data=data, files=files, headers=headers)
            print(f"API Response: {res.status_code}")
        except Exception as e:
            print(f"Error sending data to server: {e}")