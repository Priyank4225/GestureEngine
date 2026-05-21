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
        input_data = landmarks.flatten().reshape(-1, 1)
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