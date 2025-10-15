# Sign Language Detection and Recognition System
A Python-powered project for phrase-level sign language gesture recognition using computer vision and machine learning. 
This tool enables automatic recognition of sign language gestures from images or webcam video, converting them to text for increased accessibility.
<img src="https://github.com/user-attachments/assets/5acdb601-d6d1-4e6d-b526-685e0517d13c" width="800" /> 
<img src="https://github.com/user-attachments/assets/4d42bde3-474a-48c6-b8cb-674ca6c21d11" width="800" />
<img src="https://github.com/user-attachments/assets/eff0317b-827f-4261-aaf1-0acdb9a8fd99" width="600" /> 
<img src="https://github.com/user-attachments/assets/d697886f-b1fd-472f-b588-39c3da447e6a" width="800" /> 
<img src="https://github.com/user-attachments/assets/90d6d1c3-62b9-4640-ab12-572571b163d6" width="800" />
Sign language is a critical communication medium for the deaf and hard-of-hearing community. This project creates an automated sign language detection and recognition system using cutting-edge computer vision and machine learning methods.
The system uses Mediapipe to extract 21 hand landmarks (keypoints) from images or video frames, capturing essential hand posture and movement data to differentiate gestures effectively.

For temporal gesture sequences, the system employs an LSTM neural network to learn and classify phrase-level sign language with high accuracy. For static gesture images, dense neural networks are trained for robust classification.
In real-time scenarios, OpenCV is used to access webcam streams, detect hand gestures frame-by-frame, and display the recognized sign information directly over the video feed for clear user feedback.

The project is modular, with separate scripts handling data preprocessing, keypoint extraction, model training, and real-time prediction, allowing easy extension to new datasets or sign languages.
This system lays the groundwork for assistive communication tools and educational resources, improving accessibility and inclusion for sign language users.

By integrating AI and visual recognition techniques, this initiative bridges the gap between signed gestures and automatic recognition, enabling smoother communication in everyday contexts.

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
