import cv2
import face_recognition
import numpy as np
import json
import os


FACE_DATA_FILE = "face_data.json"


def load_face_data():
    if os.path.exists(FACE_DATA_FILE) and os.path.getsize(FACE_DATA_FILE) > 0:
        try:
            with open(FACE_DATA_FILE, 'r') as file:
                face_data = json.load(file)
                for name, encoding in face_data.items():
                    face_data[name] = np.array(encoding)
                return face_data
        except json.JSONDecodeError:
            print("Error decoding JSON. The file might be corrupted.")
            return {}
    else:
        return {}

def save_face_data(face_data):

    face_data_to_save = {}

    for name, encoding in face_data.items():
        face_data_to_save[name] = encoding.tolist()

    try:
        with open(FACE_DATA_FILE, 'w') as file:
            json.dump(face_data_to_save, file)
    except Exception as e:
        print(f"Error saving face data: {e}")



def find_matching_face(face_encoding, face_data):
    for name, known_encoding in face_data.items():
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        if True in matches:
            return name
    return None

def process_frame(frame, face_data, save_new_face=False):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = find_matching_face(face_encoding, face_data)

        if save_new_face:
            if name is None:
                print("Face not found in database.")
                name = input("Enter the name of the person: ")
                face_data[name] = face_encoding
                save_face_data(face_data)
                print(f"Face data for {name} saved.")
            else:
                print(f"Face already exists in database as {name}.")
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        if name:
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

face_data = load_face_data()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

 
    save_new_face = False


    if cv2.waitKey(1) & 0xFF == ord('s'):
        save_new_face = True


    frame = process_frame(frame, face_data, save_new_face)


    cv2.imshow("Video", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
