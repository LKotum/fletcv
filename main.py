import flet as ft
import cv2
import dlib
import numpy as np
import sqlite3
import time
import threading
import os
import platform
import base64
from queue import Queue
from typing import Optional

# Set environment variable for macOS camera permissions
if platform.system() == "Darwin":
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

# Constants
DATABASE_NAME = "faces.db"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DLIB_SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

class FaceRecognitionApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Face Recognition App"
        self.page.window_width = 1000
        self.page.window_height = 700
        self.page.window_resizable = False  # Fix window size

        # Initialize black frame first
        self.black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, img_encoded = cv2.imencode('.jpg', self.black_frame)
        self.black_frame_bytes = base64.b64encode(img_encoded).decode("utf-8")

        # Camera variables
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_processing = False
        self.current_camera_index: Optional[int] = None

        # Face detection models
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR_PATH)
        except:
            self.predictor = None
            print("Warning: Could not load dlib shape predictor.")

        # Database setup
        self.db_queue = Queue()
        self.setup_database()

        # UI elements
        self.create_ui()

        # Camera thread
        self.camera_thread = None

        # Start database worker thread
        self.db_worker_thread = threading.Thread(target=self.db_worker, daemon=True)
        self.db_worker_thread.start()

    def db_worker(self):
        """Worker thread to handle all database operations"""
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        while True:
            task = self.db_queue.get()
            if task is None:  # Sentinel value to stop the thread
                break

            func, args = task
            try:
                func(cursor, *args)
            except Exception as e:
                print(f"Database error: {e}")
            finally:
                self.db_queue.task_done()

        conn.close()

    def db_execute(self, func, *args):
        """Execute a database function in the worker thread"""
        self.db_queue.put((func, args))

    def setup_database(self):
        def init_db(cursor):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    descriptor BLOB
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_landmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id INTEGER,
                    landmark_index INTEGER,
                    x REAL,
                    y REAL,
                    FOREIGN KEY (face_id) REFERENCES faces(id)
                )
            """)
            cursor.connection.commit()

        self.db_execute(init_db)

    def create_ui(self):
        # Camera selection dropdown
        self.camera_dropdown = ft.Dropdown(
            label="Select Camera",
            options=[ft.dropdown.Option("Select device")],
            on_change=self.on_camera_selected,
            width=300
        )

        # Buttons
        self.start_button = ft.ElevatedButton(
            text="Start",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self.start_processing,
            disabled=True
        )

        self.stop_button = ft.ElevatedButton(
            text="Stop",
            icon=ft.Icons.STOP,
            on_click=self.stop_processing,
            disabled=True
        )

        # Image display
        self.camera_image = ft.Image(
            width=800,
            height=600,
            fit=ft.ImageFit.CONTAIN,
            src_base64=self.black_frame_bytes  # Use the initialized black frame
        )

        # Status text
        self.status_text = ft.Text("Select a camera and press Start")

        # Assemble the UI
        self.page.add(
            ft.Row(
                controls=[
                    self.camera_dropdown,
                    self.start_button,
                    self.stop_button
                ],
                alignment=ft.MainAxisAlignment.START
            ),
            ft.Divider(),
            self.camera_image,
            self.status_text
        )

        # Populate camera list after UI is created
        self.refresh_camera_list()

    def refresh_camera_list(self):
        # Clear existing options (keep the first "Select device" option)
        self.camera_dropdown.options = [ft.dropdown.Option("Select device")]

        # Test cameras with better error handling
        available_cameras = self.detect_available_cameras()

        # Add available cameras to dropdown
        for cam_idx in available_cameras:
            self.camera_dropdown.options.append(
                ft.dropdown.Option(text=f"Camera {cam_idx}", key=str(cam_idx))
            )

        self.page.update()

    def detect_available_cameras(self, max_cameras=3):
        available = []
        for i in range(max_cameras):
            cap = None
            try:
                backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened() and cap.read()[0]:
                    available.append(i)
            except Exception as e:
                print(f"Error checking camera {i}: {str(e)}")
            finally:
                if cap is not None:
                    cap.release()
        return available

    def on_camera_selected(self, e):
        if self.is_processing:
            self.stop_processing(None)

        # Show black frame when switching cameras
        self.camera_image.src_base64 = self.black_frame_bytes
        self.page.update()

        if self.camera_dropdown.value == "Select device":
            self.start_button.disabled = True
            self.stop_button.disabled = True
        else:
            self.start_button.disabled = False
            self.stop_button.disabled = True

        self.page.update()

    def start_processing(self, e):
        if self.camera_dropdown.value == "Select device":
            return

        self.current_camera_index = int(self.camera_dropdown.value)
        self.start_button.disabled = True
        self.stop_button.disabled = False
        self.status_text.value = f"Processing camera {self.current_camera_index}..."
        self.page.update()

        self.is_processing = True
        self.camera_thread = threading.Thread(target=self.process_camera_feed, daemon=True)
        self.camera_thread.start()

    def stop_processing(self, e):
        self.is_processing = False
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.status_text.value = "Processing stopped"

        # Show black frame when stopping
        self.camera_image.src_base64 = self.black_frame_bytes
        self.page.update()

        if self.capture and self.capture.isOpened():
            self.capture.release()
            self.capture = None

    def process_camera_feed(self):
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(self.current_camera_index, backend)

        if not self.capture.isOpened():
            self.status_text.value = "Error: Could not open camera"
            self.page.update()
            return

        while self.is_processing and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            _, img_encoded = cv2.imencode('.jpg', processed_frame)
            img_bytes = img_encoded.tobytes()

            self.camera_image.src_base64 = base64.b64encode(img_bytes).decode("utf-8")
            self.page.update()
            time.sleep(0.03)

        if self.capture and self.capture.isOpened():
            self.capture.release()

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if self.predictor:
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                landmarks = self.predictor(gray, dlib_rect)

                # Check if face is new in the database
                face_id = self.check_face_in_db(landmarks)

                if face_id is not None:
                    cv2.putText(frame, f"New Face #{face_id}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    # Find the existing face ID that matches
                    matched_id = self.find_matching_face(landmarks)
                    if matched_id:
                        cv2.putText(frame, f"Face #{matched_id}", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw landmarks
                for i in range(landmarks.num_parts):
                    point = landmarks.part(i)
                    cv2.circle(frame, (point.x, point.y), 2, (0, 0, 255), -1)

        return frame

    def find_matching_face(self, landmarks):
        """Find the existing face ID that matches the landmarks"""
        landmarks_array = np.array([[point.x, point.y] for point in landmarks.parts()])
        mean = np.mean(landmarks_array, axis=0)
        std = np.std(landmarks_array, axis=0)
        normalized_landmarks = (landmarks_array - mean) / std

        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT id, descriptor FROM faces")
            faces = cursor.fetchall()

            for face_id, descriptor_bytes in faces:
                db_landmarks = np.frombuffer(descriptor_bytes, dtype=np.float64)
                db_landmarks = db_landmarks.reshape(-1, 2)
                distance = np.linalg.norm(normalized_landmarks - db_landmarks)
                if distance < 10.0:
                    return face_id
            return None
        finally:
            conn.close()

    def check_face_in_db(self, landmarks):
        """Check if face exists in database and return face_id if new"""
        landmarks_array = np.array([[point.x, point.y] for point in landmarks.parts()])
        mean = np.mean(landmarks_array, axis=0)
        std = np.std(landmarks_array, axis=0)
        normalized_landmarks = (landmarks_array - mean) / std

        # We need to run this synchronously since we need the result immediately
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT id, descriptor FROM faces")
            faces = cursor.fetchall()

            if not faces:
                # Store new face
                descriptor_bytes = normalized_landmarks.tobytes()
                cursor.execute(
                    "INSERT INTO faces (descriptor) VALUES (?)",
                    (descriptor_bytes,)
                )
                face_id = cursor.lastrowid

                # Store landmarks
                for i, (x, y) in enumerate(landmarks_array):
                    cursor.execute(
                        "INSERT INTO face_landmarks (face_id, landmark_index, x, y) VALUES (?, ?, ?, ?)",
                        (face_id, i, x, y)
                    )

                conn.commit()
                return face_id

            for face_id, descriptor_bytes in faces:
                db_landmarks = np.frombuffer(descriptor_bytes, dtype=np.float64)
                db_landmarks = db_landmarks.reshape(-1, 2)
                distance = np.linalg.norm(normalized_landmarks - db_landmarks)
                if distance < 10.0:
                    return None  # Face exists

            # If we get here, face is new
            descriptor_bytes = normalized_landmarks.tobytes()
            cursor.execute(
                "INSERT INTO faces (descriptor) VALUES (?)",
                (descriptor_bytes,)
            )
            face_id = cursor.lastrowid

            # Store landmarks
            for i, (x, y) in enumerate(landmarks_array):
                cursor.execute(
                    "INSERT INTO face_landmarks (face_id, landmark_index, x, y) VALUES (?, ?, ?, ?)",
                    (face_id, i, x, y)
                )

            conn.commit()
            return face_id
        finally:
            conn.close()

    def __del__(self):
        """Clean up when the app is closed"""
        if hasattr(self, 'db_queue'):
            self.db_queue.put(None)  # Signal worker thread to exit
            if hasattr(self, 'db_worker_thread'):
                self.db_worker_thread.join()

def main(page: ft.Page):
    FaceRecognitionApp(page)

if __name__ == "__main__":
    ft.app(target=main)