import cv2
import tensorflow as tf
import numpy as np
import random
from PIL import Image

# Load Face Detector and ML Model
face_classifier = cv2.CascadeClassifier('./Harcascade/haarcascade_frontalface_default.xml')
classifier = tf.keras.models.load_model('./Models/model_v_47.hdf5')

# Emotion Labels
image_class_labels = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "neutral": "😐", "sad": "😔", "surprise": "😮", "no face": "😐"
}

# Recommendations
recommendation_text = {
    "anger": ["Take a deep breath.", "Try to calm down.", "Focus on something positive."],
    "happy": ["Keep smiling!", "Enjoy the moment!", "Spread happiness!"],
    "sad": ["It's okay to be sad.", "Reach out to a friend.", "Better days are ahead!"],
    "neutral": ["Maybe try doing something fun!", "Enjoy your peace.", "Think positive."],
    "fear": ["Face your fears slowly.", "Take deep breaths.", "You're stronger than your fears."],
    "disgust": ["Focus on something refreshing.", "Let go of negativity.", "Think about happy memories."],
    "surprise": ["Embrace the unexpected!", "Life is full of surprises!", "Enjoy the moment!"],
    "no face": ["Ensure proper lighting.", "Position your face correctly.", "Try again with better visibility."]
}

# Capture Image from Webcam
def capture_image():
    cap = cv2.VideoCapture(0)  # Open webcam
    print("Press 'SPACE' to capture an image...")

    while True:
        ret, frame = cap.read()
        cv2.imshow('Press SPACE to Capture', frame)

        key = cv2.waitKey(1)
        if key == 32:  # Space key to capture
            image = frame
            break
        elif key == 27:  # ESC key to exit
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return image

# Process Image for Emotion Detection
def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "no face"

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        face_roi = face_roi.astype("float") / 255.0
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)

        prediction = classifier.predict(face_roi)[0]
        predicted_label = np.argmax(prediction)

        return image_class_labels[predicted_label]

# Main Execution
if __name__ == "__main__":
    captured_image = capture_image()

    if captured_image is not None:
        emotion = predict_emotion(captured_image)
        print(f"Detected Emotion: {emotion} {emotions_emoji_dict[emotion]}")
        print(f"Suggestion: {random.choice(recommendation_text[emotion])}")
    else:
        print("No image captured.")
