import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and label encoder
model = load_model('sign_language_model.h5')
label_classes = np.load('label_classes.npy', allow_pickle=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Set max_seq_len to the value used during model training
max_seq_len = model.input_shape[1]

cap = cv2.VideoCapture(0)

frame_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: reduce frame size for faster processing
    frame = cv2.resize(frame, (1280, 720))  # HD size
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        frame_buffer.append(keypoints)
    else:
        frame_buffer.append([0.0]*63)  # If no hand detected, pad with zeros

    # Only keep the latest max_seq_len frames
    if len(frame_buffer) > max_seq_len:
        frame_buffer = frame_buffer[-max_seq_len:]

    # Once buffer is full, make prediction every frame
    if len(frame_buffer) == max_seq_len:
        input_sequence = np.expand_dims(np.array(frame_buffer), axis=0)  # shape: (1, max_seq_len, 63)
        prediction = model.predict(input_sequence, verbose=0)
        predicted_class = np.argmax(prediction)
        predicted_label = str(label_classes[predicted_class])

        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
