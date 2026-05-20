import numpy as np
import json

model = "ThumbsUp"
with open(f"Models/{model}/model.json", "r") as f:
    datapoints = json.load(f)["labels"]
data = np.load(f"Models/{model}/model_weights.npz")
layer1_w = data["W1"]
layer2_w = data["W2"]
layer3_w = data["W3"]
layer1_b = data["b1"]
layer2_b = data["b2"]
layer3_b = data["b3"]

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def get_pred(landmarks, accuracy=0.70):
    hand_landmarks_list = landmarks.hand_landmarks
    result = []
    for hand_landmarks in hand_landmarks_list:
        input_data = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks]
        input_data = np.array(input_data).flatten().reshape(-1, 1)
        l1_out = ReLU(layer1_w.dot(input_data) + layer1_b)
        l2_out = ReLU(layer2_w.dot(l1_out) + layer2_b)
        l3_out = layer3_w.dot(l2_out) + layer3_b
        output = softmax(l3_out)
        pred_idx = np.argmax(output)
        if output[pred_idx][0] >= accuracy:
            result.append(datapoints[pred_idx])
        else:
            result.append(datapoints[-1])
    return result