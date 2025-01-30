import os
import cv2
import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import random
import joblib
import tensorflow as tf
# from tf.keras.models import load_model
# from keras.api.preprocessing.image import img_to_array
from track_utils import add_prediction_details, create_emotionclf_table, IST, view_all_prediction_details  # Import IST from track_utils

# Load Model
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

# IMAGE Model
face_classifier = cv2.CascadeClassifier('./Harcascade/haarcascade_frontalface_default.xml')
classifier=tf.keras.models.load_model('./Models/model_v_47.hdf5')
image_class_labels={0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


recommendation_text = {
    "anger": [
        "Hey, take a deep breath and relax.",
        "It's okay to be upset, but don't let anger control you.",
        "Try to channel your anger into something productive.",
        "Let go of the frustration and focus on peace.",
        "Take a step back and cool down, you've got this!"
    ],
    "joy": [
        "Hey, keep spreading that happiness!",
        "Your joy is contagious, keep it up!",
        "Enjoy every moment and stay cheerful.",
        "Stay positive and keep shining!",
        "Happiness looks great on you!"
    ],
    "sadness": [
        "Hey, it's okay to feel sad sometimes.",
        "Remember, tough times don't last forever.",
        "You're stronger than you think, keep pushing forward.",
        "Surround yourself with things that make you happy.",
        "If you need to talk, I'm here for you."
    ],
    "disgust": [
        "Hey, try to focus on something positive.",
        "Disgust is temporary, let it pass.",
        "Clear your mind and think of something refreshing.",
        "Don't let negative feelings ruin your day.",
        "Find something that brings you joy instead!"
    ],
    "fear": [
        "Hey, you're stronger than your fears.",
        "Take deep breaths, you're safe.",
        "Courage is not the absence of fear, but moving forward despite it.",
        "You're not alone; you can overcome this.",
        "Face your fears one step at a time."
    ],
    "happy": [
        "Hey, keep enjoying the good moments!",
        "Happiness suits you well!",
        "Stay joyful and keep smiling!",
        "You're radiating positive energy!",
        "Keep spreading the happiness!"
    ],
    "neutral": [
        "Hey, feeling neutral is okay too.",
        "It's a calm moment, enjoy it.",
        "Maybe try doing something exciting?",
        "Find something that sparks your interest!",
        "Being neutral is fine, but don't stay stuck!"
    ],
    "sad": [
        "Hey, better days are coming.",
        "You're not alone, reach out to someone.",
        "Sadness is temporary, you will smile again.",
        "Take it easy and be kind to yourself.",
        "Remember, after every storm comes a rainbow."
    ],
    "shame": [
        "Hey, don't be too hard on yourself.",
        "We all make mistakes, it's okay.",
        "Learn from it and move forward.",
        "You're worthy and valuable, no matter what.",
        "Self-forgiveness is the first step to healing."
    ],
    "surprise": [
        "Hey, life is full of surprises!",
        "Enjoy the unexpected moments!",
        "Surprise keeps life interesting!",
        "Take the moment in and embrace it.",
        "Sometimes surprises lead to great things!"
    ],
    "no face": [
        "Click an appropriate picture.",
        "Make sure your face is visible.",
        "Try again with better lighting.",
        "Position your face properly for detection.",
        "Make sure your camera is capturing your face."
    ]
}

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮", "no face": "😐"}

# Main Application
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    create_emotionclf_table()
    if choice == "Home":
        st.subheader("Emotion Detection in Text")

        with st.form(key='emotion_clf_form'):
            name = st.text_input("Enter your Name")
            age = st.text_input("Enter your Age")
            primary_hobby = st.text_input("Enter a Primary Hobby")
            raw_text = st.text_area("Type for emotion detection")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text and len(raw_text):
            col1, col2 = st.columns(2)


            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            print(age, name, primary_hobby)

            add_prediction_details(age, name, primary_hobby, raw_text, prediction, np.max(probability), datetime.now(IST))

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                recommendation = recommendation_text[prediction]
                st.write("{}: {}".format(prediction.capitalize(), emoji_icon))
                st.write("Confidence: {}".format(np.max(probability)))
                st.write("Suggestion: {}".format(recommendation[random.randint(0, len(recommendation)-1)]))

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

        st.subheader("Emotion Detection in Images")

        with st.form(key='emotion_clf_image_form'):
            raw_image = st.camera_input("Take a picture")
            submit_image = st.form_submit_button(label='Submit')

        if submit_image and raw_image:     
            col1, col2 = st.columns(2)       
            # Decode the BytesIO object to a NumPy array
            image = Image.open(raw_image)
            img = np.array(image)

            # Convert to BGR format (if needed, since OpenCV expects BGR, but PIL gives RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            allfaces = []
            rects = []

            # Process detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                allfaces.append(roi_gray)
                rects.append((x, w, y, h))
            
            # print(allfaces)
            
            with col1:
                st.success("Original Image")
                st.image(gray, caption="Original Image")

            with col2:
                image_prediction = "no face"
                i = 0
                for face in allfaces:
                    roi = face.astype("float") / 255.0
                    # roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=-1)
                    roi = np.expand_dims(roi, axis=0)
                    print(f"ROI shape: {roi.shape}")    # 1,48,48,x
                    print(f"Expected input shape: {classifier.input_shape}") # None, 48,48,x
                    img_prediction = classifier.predict(roi)[0]
                    print(img_prediction, type(img_prediction))
                    img_prediction = np.where(img_prediction == max(img_prediction))
                    print(img_prediction[0][0])
                    image_prediction = image_class_labels[img_prediction[0][0]]  # Replace 'classifier' with your loaded model
                    st.write(f"Prediction for face {i+1}: {image_prediction}")
                    i += 1
                    
                st.success("Prediction")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Processed Image")
                emoji_icon = emotions_emoji_dict[image_prediction]
                st.write("{}: {}".format(image_prediction, emoji_icon))
                recommendation = recommendation_text[image_prediction]
                st.write("Suggestion: {}".format(recommendation[random.randint(0, len(recommendation)-1)]))

        st.subheader("Emotion Detection in Text+Images")
        
        with st.form(key='emotion_clf_combination_image_form'):
            name_combination = st.text_input("Enter your Name")
            age_combination = st.text_input("Enter your Age")
            primary_hobby_combination = st.text_input("Enter a Primary Hobby")
            raw_text_combination = st.text_area("Type for combination emotion detection")
            raw_image_combination = st.camera_input("Take a picture")
            submit_form = st.form_submit_button(label='Submit')
        
        if submit_form and len(raw_text_combination):

            col1, col2 = st.columns(2)

            prediction_combination = predict_emotions(raw_text_combination)
            probability_combination = get_prediction_proba(raw_text_combination)

            add_prediction_details(age_combination, name_combination, primary_hobby_combination, raw_text_combination, prediction_combination, np.max(probability_combination), datetime.now(IST))

            image_combination = Image.open(raw_image_combination)
            img_combination = np.array(image_combination)

            # Convert to BGR format (if needed, since OpenCV expects BGR, but PIL gives RGB)
            img_combination = cv2.cvtColor(img_combination, cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray_combination = cv2.cvtColor(img_combination, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces_combination = face_classifier.detectMultiScale(gray_combination, 1.3, 5)
            allfaces_combination = []
            rects_combination = []

            # Process detected faces
            for (x, y, w, h) in faces_combination:
                cv2.rectangle(img_combination, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray_combination = gray_combination[y:y+h, x:x+w]
                roi_gray_combination = cv2.resize(roi_gray_combination, (48, 48), interpolation=cv2.INTER_AREA)
                allfaces_combination.append(roi_gray_combination)
                rects_combination.append((x, w, y, h))

            with col1:
                st.success("Original Text")
                st.write(raw_text_combination)

                st.success("Prediction")
                emoji_icon_combination = emotions_emoji_dict[prediction_combination]
                recommendation_combination = recommendation_text[prediction_combination]
                st.write("{}: {}".format(prediction_combination.capitalize(), emoji_icon_combination))
                st.write("Confidence: {}".format(np.max(probability_combination)))
                # st.write("Suggestion: {}".format(recommendation_combination[random.randint(0, len(recommendation_combination)-1)]))
            
                st.success("Original Image")
                st.image(gray_combination, caption="Original Image")

            with col2:
                st.success("Prediction Probability")
                proba_df_combination = pd.DataFrame(probability_combination, columns=pipe_lr.classes_)
                proba_df_clean_combination = proba_df_combination.T.reset_index()
                proba_df_clean_combination.columns = ["emotions", "probability"]

                fig_combination = alt.Chart(proba_df_clean_combination).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig_combination, use_container_width=True)

                image_prediction_combination = "no face"
                i = 0
                for face in allfaces_combination:
                    roi_combination = face.astype("float") / 255.0
                    # roi = img_to_array(roi)
                    roi_combination = np.expand_dims(roi_combination, axis=-1)
                    roi_combination = np.expand_dims(roi_combination, axis=0)
                    print(f"ROI shape: {roi_combination.shape}")    # 1,48,48,x
                    print(f"Expected input shape: {classifier.input_shape}") # None, 48,48,x
                    img_prediction_combination = classifier.predict(roi)[0]
                    print(img_prediction_combination, type(img_prediction_combination))
                    img_prediction_combination = np.where(img_prediction_combination == max(img_prediction_combination))
                    print(img_prediction_combination[0][0])
                    image_prediction_combination = image_class_labels[img_prediction_combination[0][0]]  # Replace 'classifier' with your loaded model
                    st.write(f"Prediction for face {i+1}: {image_prediction_combination}")
                    i += 1
                    
                st.success("Prediction")
                st.image(cv2.cvtColor(img_combination, cv2.COLOR_BGR2RGB), caption="Processed Image")
                emoji_icon_combination = emotions_emoji_dict[image_prediction_combination]
                st.write("{}: {}".format(image_prediction_combination, emoji_icon_combination))
                recommendation_combination = recommendation_text[image_prediction_combination]
                if prediction_combination != image_prediction_combination and image_prediction_combination != "no face":
                    st.write(f"You sound {prediction_combination} but you look {image_prediction_combination}")

                if image_prediction_combination == "no face":
                    recommendation_combination = recommendation_text[prediction_combination]
                st.write("Suggestion: {}".format(recommendation_combination[random.randint(0, len(recommendation_combination)-1)]))


    elif choice == "Monitor":
        st.subheader("Monitor App")

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=["Age", "Name", "Primary Hobby", 'Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)

    elif choice == "About":
        st.write("Welcome to the Emotion Detection in Text App! This application utilizes the power of natural language processing and machine learning to analyze and identify emotions in textual data.")

        st.subheader("Our Mission")

        st.write("At Emotion Detection in Text, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text. We believe that emotions play a crucial role in communication, and by uncovering these emotions, we can gain valuable insights into the underlying sentiments and attitudes expressed in written text.")

        st.subheader("How It Works")

        st.write("When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text. These features are then fed into the trained model, which predicts the emotions associated with the input text. The app displays the detected emotions, along with a confidence score, providing you with valuable insights into the emotional content of your text.")

        st.subheader("Key Features:")

        st.markdown("##### 1. Real-time Emotion Detection")

        st.write("Our app offers real-time emotion detection, allowing you to instantly analyze the emotions expressed in any given text. Whether you're analyzing customer feedback, social media posts, or any other form of text, our app provides you with immediate insights into the emotions underlying the text.")

        st.markdown("##### 2. Confidence Score")

        st.write("Alongside the detected emotions, our app provides a confidence score, indicating the model's certainty in its predictions. This score helps you gauge the reliability of the emotion detection results and make more informed decisions based on the analysis.")

        st.markdown("##### 3. User-friendly Interface")

        st.write("We've designed our app with simplicity and usability in mind. The intuitive user interface allows you to effortlessly input text, view the results, and interpret the emotions detected. Whether you're a seasoned data scientist or someone with limited technical expertise, our app is accessible to all.")

        st.subheader("Applications")

        st.markdown("""
          The Emotion Detection in Text App has a wide range of applications across various industries and domains. Some common use cases include:
          - Social media sentiment analysis
          - Customer feedback analysis
          - Market research and consumer insights
          - Brand monitoring and reputation management
          - Content analysis and recommendation systems
          """)


if __name__ == '__main__':
    main()
