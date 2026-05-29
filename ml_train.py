import numpy as np
import pandas as pd
import json
from tools import *

model = "ThumbsUp"
data = pd.read_csv(f'Models/{model}/data.csv')
with open(f"Models/{model}/model.json", "r") as f:
    file = json.load(f)
    datapoints = file["labels"]
    desc = file["dims"]
layers = len(desc)-1

data = np.array(data)
n = data.shape[1]
np.random.shuffle(data)

data_train = data.T
Y_train = data_train[0]
Y_train = Y_train.astype(int)
X_train = data_train[1:n]

def init_params():
    weights = []
    biases = []
    for i in range(layers):
        weights.append(np.random.randn(desc[i+1], desc[i]) * np.sqrt(2 / desc[i]))
        biases.append(np.zeros((desc[i+1], 1)))
    return weights, biases

def forward_prop(weights, biases, X):
    res = X
    Z = []
    A = []
    for i in range(layers):
        Z.append(weights[i].dot(res) + biases[i])
        if i == layers-1:
            res = softmax(Z[-1])
        else:
            res = ReLU(Z[-1])
        A.append(res)
    return Z, A

def backward_prop(weights, Z, A, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    db = [0]*layers
    dW = [0]*layers
    for i in range(layers-1, -1, -1):
        if i == layers-1:
            dZ = A[i] - one_hot_Y
        else:
            dZ = weights[i+1].T.dot(dZ) * ReLU_deriv(Z[i])
        if i == 0:
            dW[i] = 1 / m * dZ.dot(X.T)
            pass
        else:
            dW[i] = 1 / m * dZ.dot(A[i-1].T)
        db[i] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    return dW, db

def update_params(weights, biases, dW, db, alpha):
    for i in range(layers):
        weights[i] -= alpha * dW[i]
        biases[i] -= alpha * db[i]
    return weights, biases

def gradient_descent(X, Y, alpha, iterations):
    weights, biases = init_params()
    for i in range(iterations):
        Z, A = forward_prop(weights, biases, X)
        dW, db = backward_prop(weights, Z, A, X, Y)
        weights, biases = update_params(weights, biases, dW, db, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A[-1])
            print(get_accuracy(predictions, Y))
    return weights, biases

weights, biases = gradient_descent(X_train, Y_train, 0.10, 500)
save_weights = {}
save_biases = {}
for i in range(layers):
    save_weights[f"W{i+1}"] = weights[i]
    save_biases[f"b{i+1}"] = biases[i]
np.savez(f"Models/{model}/model_weights.npz", **save_weights)
np.savez(f"Models/{model}/model_biases.npz", **save_biases)