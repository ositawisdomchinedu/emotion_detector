# Emotion Detector

A real-time facial emotion recognition application built with Deep Learning, TensorFlow, and OpenCV. The application uses your webcam to detect faces and classify facial expressions into one of seven common emotions in real time.

## Overview

This project demonstrates the practical application of Convolutional Neural Networks (CNNs) for computer vision. A custom CNN model was trained on the FER-2013 dataset and integrated with OpenCV to perform live facial emotion detection from a webcam feed.

The detected face is highlighted with a bounding box, and the predicted emotion is displayed instantly.

## Features

- Real-time webcam emotion detection
- Face detection using OpenCV
- CNN-based emotion classification
- Supports seven facial emotions:
  - 😠 Angry
  - 🤢 Disgust
  - 😨 Fear
  - 😊 Happy
  - 😐 Neutral
  - 😢 Sad
  - 😲 Surprise
- Live prediction with confidence-based classification

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

## Dataset

The model was trained using the **FER-2013 (Facial Expression Recognition 2013)** dataset, which contains approximately 35,000 grayscale facial images labeled across seven emotion categories.

The dataset can be downloaded from Kaggle.

## Project Structure

```text
emotion-detector/
│
├── data/
├── models/
│   └── emotion_model.h5
├── train_model.py
├── detect.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ositawisdomchinedu/emotion_detector.git
cd emotion_detector
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a requirements file:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

For Apple Silicon (M1/M2/M3):

```bash
pip install tensorflow-macos tensorflow-metal opencv-python numpy matplotlib
```

## Training the Model

1. Download the FER-2013 dataset.
2. Place the dataset inside the `data/` directory.
3. Train the CNN model:

```bash
python train_model.py
```

After training completes, the model will be saved as:

```text
emotion_model.h5
```

If the pretrained model is already included, you can skip this step.

## Running the Application

Launch the real-time emotion detector:

```bash
python detect.py
```

The webcam will open automatically and begin detecting facial emotions.

Press **Q** at any time to close the application.

## Model Performance

The model achieves approximately **60–65% validation accuracy**, due to its challenging nature and variations in lighting, facial angles, and image quality.

For better prediction results:

- Use good lighting.
- Face the camera directly.
- Avoid excessive head rotation or occlusion.

## Learning Outcomes

This project strengthened my understanding of:

- Convolutional Neural Networks (CNNs)
- Image preprocessing
- Computer Vision with OpenCV
- TensorFlow/Keras model training
- Real-time inference pipelines
- Deep Learning model deployment

## Future Improvements

- Improve model accuracy using transfer learning
- Add emotion confidence scores
- Support multiple face detection
- Deploy as a web application using Streamlit or Flask
- Optimize inference speed for edge devices

## Author

**Osita Wisdom Chinedu**

AI/ML Engineer | Data Scientist

GitHub: https://github.com/ositawisdomchinedu

---

If you find this project helpful, feel free to ⭐ the repository and contribute with suggestions or improvements.