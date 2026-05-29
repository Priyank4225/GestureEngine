import numpy as np
import json
from tools import *

model = "ThumbsUp"
with open(f"Models/{model}/model.json", "r") as f:
    file = json.load(f)
    datapoints = file["labels"]
    dims = file["dims"]
layers = len(dims)-1
data = np.load(f"Models/{model}/model_weights.npz")
weights = np.load(f"Models/{model}/model_weights.npz")
biases = np.load(f"Models/{model}/model_biases.npz")
weights = [weights[f"W{str(i+1)}"] for i in range(layers)]
biases = [biases[f"b{str(i+1)}"] for i in range(layers)]


def get_pred(detection_result, accuracy=0.70):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    result = []
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx][0].category_name
        landmarks = []
        if handedness == "Right":
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
        res = landmarks.flatten().reshape(-1, 1)
        for i in range(layers):
            if i == layers-1:
                res = softmax(weights[i].dot(res) + biases[i])
            else:
                res = ReLU(weights[i].dot(res) + biases[i])
        pred_idx = get_predictions(res)[0]
        print(pred_idx)
        if res[pred_idx][0] >= accuracy:
            result.append(datapoints[pred_idx])
        else:
            result.append(datapoints[-1])
    return result