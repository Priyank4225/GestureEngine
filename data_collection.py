import cv2
import csv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw import draw_landmarks_on_image

model = "ThumbsUp"

cam = cv2.VideoCapture(0)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)
csv_file = open(f"Models/{model}/data.csv", "a", newline="")
csv_writer = csv.writer(csv_file)
cap = None
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgRGB = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    detection_result = detector.detect(imgRGB)
    annotated_image = draw_landmarks_on_image(frame, detection_result)
    cv2.imshow("webcam", annotated_image)
    inp = cv2.waitKey(1)
    if inp == ord('q'):
        break
    elif inp == ord('c'):
        label = input("Enter label index: ")
        if input("Confirm Y/N?") == "Y":
            if len(detection_result.hand_landmarks) > 0:
                hand_landmarks = detection_result.hand_landmarks[0]
                row = [label]
                landmarks = []
                if detection_result.handedness[0][0].category_name == "Right":
                    for landmark in hand_landmarks:
                        landmarks.append([
                            -landmark.x,
                            landmark.y,
                            landmark.z
                        ])
                else:
                    for landmark in hand_landmarks:
                        landmarks.append([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])
                landmarks = np.array(landmarks)
                scale = np.sqrt(np.sum(np.square(landmarks[9] - landmarks[0])))
                landmarks = (landmarks - landmarks[0]) / scale
                row.extend(landmarks.flatten())
                csv_writer.writerow(row)
                print("Saved.")
csv_file.close()
cam.release()
cv2.destroyAllWindows()