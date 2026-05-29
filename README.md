# GestureEngine ✋⚡

GestureEngine is a real-time hand gesture recognition system built using Python, MediaPipe, OpenCV, and a custom neural network implemented entirely from scratch using NumPy.

The project provides a complete machine learning pipeline for:

- Hand landmark detection
- Dataset collection
- Landmark normalization
- Neural network training
- Real-time gesture prediction
- Live webcam visualization

Unlike many gesture recognition projects that rely entirely on high-level ML frameworks, GestureEngine implements forward propagation, backpropagation, gradient descent, and inference manually for educational and research purposes.

## Creating Your Own Gesture Model

Want to build a custom gesture recognition system? GestureEngine makes it easy.

1. Clone this repository.
2. Create a new folder inside the `Models/` directory for your model.
3. Create a `model.json` file containing your model configuration (labels, network dimensions, etc.).
4. Update the `model` variable in the project files to match your model's name.
5. Collect training data using `data_collection.py`.
6. Train the model using `ml_train.py`.
7. Run `main.py` to test your gesture recognizer in real time.

Make sure you use the same model name during data collection, training, and inference so that the correct files are loaded.

Have fun experimenting with your own gestures and models!

---

# Features

- Real-time hand tracking using MediaPipe
- Custom gesture dataset collection tool
- Hand landmark normalization for robust predictions
- Fully custom neural network implementation
- Live webcam prediction overlay
- Multi-hand support
- Modular architecture
- Lightweight and fast inference

---

# Demo Pipeline

```text
Webcam Feed
    ↓
MediaPipe Hand Landmark Detection
    ↓
Landmark Normalization
    ↓
Custom Neural Network
    ↓
Gesture Prediction
    ↓
Live Visualization
```

---

# Project Structure

```text
GestureEngine/
│
├── main.py
├── data_collection.py
├── draw.py
├── ml.py
├── ml_train.py
├── tools.py
├── requirements.txt
├── hand_landmarker.task
│
├── Models/
│   └── ThumbsUp/
│       ├── data.csv
│       ├── model.json
│       └── model_weights.npz
│
├── LICENSE
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/yourusername/GestureEngine.git
cd GestureEngine
```

---

## 2. Create a virtual environment (recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Run Real-Time Gesture Recognition

```bash
python main.py
```

Press:

- `Q` → Quit application

---

# Dataset Collection

GestureEngine includes a built-in dataset collection tool.

Run:

```bash
python data_collection.py
```

Controls:

- `C` → Capture current hand landmarks
- `Q` → Quit

When capturing:
1. Enter label index
2. Confirm with `Y`

Captured landmark data is stored in:

```text
Models/<ModelName>/data.csv
```

---

# Model Training

Train the neural network using:

```bash
python ml_train.py
```

The script:
- Loads dataset
- Performs forward propagation
- Performs backpropagation
- Updates weights using gradient descent
- Saves trained weights

Output weights are saved as:

```text
Models/<ModelName>/model_weights.npz
```

---

# Landmark Normalization

GestureEngine normalizes hand landmarks to improve robustness and generalization.

The system performs:

## Translation Normalization

Centers landmarks relative to the wrist:

```python
landmarks = landmarks - landmarks[0]
```

---

## Scale Normalization

Normalizes hand size:

```python
scale = np.sqrt(np.sum(np.square(landmarks[9] - landmarks[0])))
landmarks = landmarks / scale
```

---

## Handedness Mirroring

Right-hand coordinates are mirrored to maintain consistency between left and right hands.

---

# Neural Network Architecture

The project uses a fully connected feedforward neural network:

```text
Input Layer
    ↓
Hidden Layer (ReLU)
    ↓
Hidden Layer (ReLU)
    ↓
Output Layer (Softmax)
```

Implemented manually using NumPy:
- ReLU activation
- Softmax activation
- Cross-entropy style gradients
- Gradient descent optimization
- He initialization

No TensorFlow or PyTorch is used for training.

---

# Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas

---

# Future Improvements

Planned upgrades include:

- Dynamic gesture recognition
- Temporal sequence models (LSTM/GRU)
- GUI dashboard
- Model evaluation metrics
- Confidence visualization
- Dataset augmentation
- TensorFlow/PyTorch comparison backend
- Export to desktop/mobile applications

---

# Why GestureEngine?

GestureEngine was built to deeply understand:
- Computer vision
- Feature engineering
- Neural networks
- Real-time inference systems
- ML pipeline design

The project focuses on learning and implementation clarity rather than relying entirely on high-level ML abstractions.

---

# License

This project is open-source and available under the MIT License.

---

# Author

Created by Priyank Sharma

If you found this project useful, consider giving the repository a ⭐ on GitHub.
