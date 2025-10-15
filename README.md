# Sign Language Detection and Recognition System

A Python-powered project for phrase-level sign language gesture recognition using computer vision and machine learning. 
This tool enables automatic recognition of sign language gestures from images or webcam video, converting them to text for increased accessibility.
## Features

- Hand gesture recognition from static images or webcam stream
- Hand keypoint extraction via Mediapipe
- Neural network classification (TensorFlow / Keras)
- Real-time prediction with OpenCV visualization
- Text-to-speech output using pyttsx3
- Modular scripts for preprocessing, feature extraction, training, and inference

## Project Structure

SignLanguage/
├── dataset/ # Raw gesture images sorted by label
├── preprocessed_dataset/ # Resized and normalized images
├── keypoints_dataset/ # Extracted hand landmark keypoints
├── models/ # Trained models and encoders
├── scripts/ # Python processing and training scripts
├── README.md # Project documentation


## Installation

1. Clone repo
   
git clone https://github.com/yourusername/SignLanguage.git
cd SignLanguage


2. (Optional) Create and activate a virtual environment
python -m venv venv

Windows
venv\Scripts\activate

macOS/Linux
source venv/bin/activate


3. Install dependencies:
pip install -r requirements.txt

## Usage

- Preprocess images:
python scripts/preprocess_images.py
- Extract keypoints:
python scripts/extract_keypoints_from_images.py
- Train model:
python scripts/train_model.py
- Run real-time prediction and speech:
python scripts/real_time_predict_and_speak.py

## Dataset

Place gesture-labeled images inside the `dataset/` folder. Supported formats: jpg, png.

## Contribution

Contributions are welcome. Feel free to open issues or submit pull requests.

---
