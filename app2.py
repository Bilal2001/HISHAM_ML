import os
import cv2
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import random
import joblib
import tensorflow as tf
import sqlite3
import hashlib

from track_utils import add_image_prediction_details, add_prediction_details, add_user, check_login, create_emotionclf_table, IST, get_ages, get_names, get_user_emotions_over_time, get_user_image_emotions_over_time, view_prediction_details, view_prediction_details_images

ADMIN_NAME = "bilal"
ADMIN_PASSWORD = "1234"

# Load Models
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

# Image Model
face_classifier = cv2.CascadeClassifier('./Harcascade/haarcascade_frontalface_default.xml')
classifier = tf.keras.models.load_model('./Models/model_v_47.hdf5')
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

# Emotion Labels
image_class_labels = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮", "no face": "😐"
}

convert_pred = {
    "joy": "joyful", "disgust": "disgusted", "anger": "angry", 
    "fear": "afraid", "happy": "happy", "sad": "sad", "sadness": "sad", 
    "shame": "ashamed", "surprise": "surprised", "neutral": "neutral"
}

# ---------------- DATABASE FUNCTIONS ------------------
create_emotionclf_table()


def logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""

# ---------------- PREDICTION FUNCTIONS ----------------
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# ---------------- STREAMLIT APP ----------------
def main():
    st.title("EmotionSense")
    
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["password"] = None

    menu = ["About", "Login", "Register"]
    if st.session_state["logged_in"]:
        if "Login" in menu: menu.remove("Login")
        if "Register" in menu: menu.remove("Register")
        if "Home" not in menu:
            menu.append("Home")
        if "Review" not in menu:
            menu.append("Review")
        if st.session_state["username"] == ADMIN_NAME and st.session_state["password"] == ADMIN_PASSWORD:
            if "Monitor" not in menu: menu.append("Monitor")
        else:
            if "Monitor" in menu: menu.remove("Monitor")
    else:
        menu = ["About", "Login", "Register"]

    choice = st.sidebar.selectbox("Menu", menu)

    # LOGIN PAGE
    if choice == "Login":
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login")

        if submit_login:
            if check_login(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["password"] = password
                st.success(f"Welcome back, {username}!")
                menu += ["Home", "Review"]

                if username == ADMIN_NAME and password == ADMIN_PASSWORD:
                    menu += ["Monitor"]
                st.rerun()
            else:
                st.error("Invalid username or password")

    # REGISTER PAGE
    elif choice == "Register":
        st.subheader("Create a New Account")
        with st.form("register_form"):
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_register = st.form_submit_button("Register")

        if submit_register:
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            elif len(new_username) == 0 or len(new_password) == 0:
                st.error("Username and password cannot be empty!")
            else:
                try:
                    add_user(new_username, new_password)
                    st.success("Account created successfully! You can now login.")
                except sqlite3.IntegrityError:
                    st.error("Username already exists! Please choose another.")
    
    # ABOUT PAGE
    elif choice == "About":
        st.subheader("About EmotionSense")
        st.write("Welcome to EmotionSense! This tool uses NLP and machine learning to detect emotions in text.")
        st.subheader("Our Mission")
        st.write("We aim to provide a real-time emotion detection tool that helps users understand emotional contexts in text.")


    # LOGOUT BUTTON
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
        if st.sidebar.button("Logout"):
            logout()

            st.rerun()

    # PROTECTED PAGES
    if "logged_in" in st.session_state and st.session_state["logged_in"]:

        # HOME PAGE
        if choice == "Home":
            st.subheader("Emotion Detection in Text")

            with st.form(key='emotion_clf_form'):
                name = st.text_input("Enter your Name")
                age = st.text_input("Enter your Age")
                raw_text = st.text_area("Type for emotion detection")
                submit_text = st.form_submit_button(label='Submit')

            if submit_text and len(raw_text):
                col1, col2 = st.columns(2)


                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)
                print(age, name)

                add_prediction_details(age, st.session_state["username"], raw_text, prediction, np.max(probability), datetime.now(IST))

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
                age = st.text_input("Enter your Age")
                submit_image = st.form_submit_button(label='Submit')

            if submit_image and raw_image and age:     
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
                        add_image_prediction_details(st.session_state['username'], age, image_prediction)
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
                raw_text_combination = st.text_area("Type for combination emotion detection")
                raw_image_combination = st.camera_input("Take a picture")
                submit_form = st.form_submit_button(label='Submit')
            
            if submit_form and len(raw_text_combination):

                col1, col2 = st.columns(2)

                prediction_combination = predict_emotions(raw_text_combination)
                probability_combination = get_prediction_proba(raw_text_combination)

                add_prediction_details(age_combination, st.session_state["username"], raw_text_combination, prediction_combination, np.max(probability_combination), datetime.now(IST))

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
                        img_prediction_combination = classifier.predict(roi_combination)[0]
                        print(img_prediction_combination, type(img_prediction_combination))
                        img_prediction_combination = np.where(img_prediction_combination == max(img_prediction_combination))
                        print(img_prediction_combination[0][0])
                        image_prediction_combination = image_class_labels[img_prediction_combination[0][0]]  # Replace 'classifier' with your loaded model
                        st.write(f"Prediction for face {i+1}: {image_prediction_combination}")
                        add_image_prediction_details(st.session_state['username'], age_combination, image_prediction_combination)
                        i += 1
                        
                    st.success("Prediction")
                    st.image(cv2.cvtColor(img_combination, cv2.COLOR_BGR2RGB), caption="Processed Image")
                    emoji_icon_combination = emotions_emoji_dict[image_prediction_combination]
                    st.write("{}: {}".format(image_prediction_combination, emoji_icon_combination))
                    recommendation_combination = recommendation_text[image_prediction_combination]
                    if prediction_combination != image_prediction_combination and image_prediction_combination != "no face":
                        st.write(f"You sound {convert_pred[prediction_combination]} but you look {convert_pred[image_prediction_combination]}")

                    if image_prediction_combination == "no face":
                        recommendation_combination = recommendation_text[prediction_combination]
                    st.write("Suggestion: {}".format(recommendation_combination[random.randint(0, len(recommendation_combination)-1)]))


        # REVIEW PAGE
        elif choice == "Review":
            st.subheader("Your Emotion Trends Over Time")
    
            # Fetch user's emotion history
            user_emotions = get_user_emotions_over_time(st.session_state["username"])
            user_image_emotions = get_user_image_emotions_over_time(st.session_state["username"])
            
            if user_emotions or user_image_emotions:
                tab1, tab2 = st.tabs(["Text Emotions", "Image Emotions"])
                
                with tab1:
                    if user_emotions:
                        # Convert to DataFrame for easier manipulation
                        df_time = pd.DataFrame(user_emotions, columns=["Username", "Emotion", "Timestamp"])
                        df_time['Timestamp'] = pd.to_datetime(df_time['Timestamp'])
                        
                        # Group by day and emotion to count occurrences
                        df_time['Date'] = df_time['Timestamp'].dt.date
                        emotion_counts = df_time.groupby(['Date', 'Emotion']).size().reset_index(name='Count')
                        
                        # Create time-series chart
                        time_chart = alt.Chart(emotion_counts).mark_line(point=True).encode(
                            x='Date:T',
                            y='Count:Q',
                            color='Emotion:N',
                            tooltip=['Date', 'Emotion', 'Count']
                        ).properties(
                            title='Your Emotion Trends Over Time (Text)',
                            width=600,
                            height=400
                        ).interactive()
                        
                        st.altair_chart(time_chart, use_container_width=True)
                        
                        # Add a summary section
                        st.subheader("Emotion Summary")
                        most_common = df_time['Emotion'].value_counts().idxmax()
                        recent_trend = df_time.sort_values('Timestamp').tail(5)['Emotion'].value_counts().idxmax()
                        
                        st.write(f"Your most common emotion: **{most_common}**")
                        st.write(f"Your recent trend: **{recent_trend}**")
                    else:
                        st.write("No text emotion data available yet. Start using the text emotion detection to see your trends!")
                
                with tab2:
                    if user_image_emotions:
                        # Convert to DataFrame for easier manipulation
                        df_img_time = pd.DataFrame(user_image_emotions, columns=["Username", "Emotion", "Timestamp"])
                        df_img_time['Timestamp'] = pd.to_datetime(df_img_time['Timestamp'])
                        
                        # Group by day and emotion to count occurrences
                        df_img_time['Date'] = df_img_time['Timestamp'].dt.date
                        img_emotion_counts = df_img_time.groupby(['Date', 'Emotion']).size().reset_index(name='Count')
                        
                        # Create time-series chart
                        img_time_chart = alt.Chart(img_emotion_counts).mark_line(point=True).encode(
                            x='Date:T',
                            y='Count:Q',
                            color='Emotion:N',
                            tooltip=['Date', 'Emotion', 'Count']
                        ).properties(
                            title='Your Emotion Trends Over Time (Images)',
                            width=600,
                            height=400
                        ).interactive()
                        
                        st.altair_chart(img_time_chart, use_container_width=True)
                        
                        # Add a summary section
                        st.subheader("Image Emotion Summary")
                        img_most_common = df_img_time['Emotion'].value_counts().idxmax()
                        img_recent_trend = df_img_time.sort_values('Timestamp').tail(5)['Emotion'].value_counts().idxmax()
                        
                        st.write(f"Your most common facial emotion: **{img_most_common}**")
                        st.write(f"Your recent facial emotion trend: **{img_recent_trend}**")
                    else:
                        st.write("No image emotion data available yet. Start using the image emotion detection to see your trends!")
            else:
                st.info("Start using the emotion detection features to see your trends over time!")
                
        # MONITOR PAGE
        elif choice == "Monitor" and st.session_state["username"] == ADMIN_NAME and st.session_state["password"] == ADMIN_PASSWORD:
            st.subheader("Admin Utilities")

            names = ["all"] + get_names()
            ages = ["all"] + get_ages()
            selected_name = "all"
            selected_age = "all"
            
            with st.expander("Emotion Classifier Metrics"):
                selected_name = st.selectbox(label="Names", options=names, placeholder="Select a Name")
                selected_age = st.selectbox(label="Ages", options=ages, placeholder="Select an Age")

                df_emotions = pd.DataFrame(view_prediction_details(selected_name, selected_age), columns=["Age", "Name", 'Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
                st.dataframe(df_emotions)
                prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
                pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
                alt.Chart(prediction_count).mark_line()
                st.altair_chart(pc, use_container_width=True)

                df_emotions_img = pd.DataFrame(view_prediction_details_images(selected_name, selected_age), columns=["Name", "Age", 'Prediction', 'Time_of_Visit'])
                st.dataframe(df_emotions_img)
                prediction_count_img = df_emotions_img['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
                pc_img = alt.Chart(prediction_count_img).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
                st.altair_chart(pc_img, use_container_width=True)

                if selected_name != "all":
                    all_emotion_data = get_user_emotions_over_time(selected_name)
                    all_image_emotion_data = get_user_image_emotions_over_time(selected_name)
                    
                    if all_emotion_data or all_image_emotion_data:
                        admin_tab1, admin_tab2 = st.tabs(["Text Emotions Trends", "Image Emotions Trends"])
                        
                        with admin_tab1:
                            if all_emotion_data:
                                df_admin_time = pd.DataFrame(all_emotion_data, columns=["Username", "Emotion", "Timestamp"])
                                df_admin_time['Timestamp'] = pd.to_datetime(df_admin_time['Timestamp'])
                                df_admin_time['Date'] = df_admin_time['Timestamp'].dt.date
                                
                                # Group by day and emotion to count occurrences
                                admin_emotion_counts = df_admin_time.groupby(['Date', 'Emotion']).size().reset_index(name='Count')
                                
                                # Create time-series chart
                                admin_time_chart = alt.Chart(admin_emotion_counts).mark_line(point=True).encode(
                                    x='Date:T',
                                    y='Count:Q',
                                    color='Emotion:N',
                                    tooltip=['Date', 'Emotion', 'Count']
                                ).properties(
                                    title=f'Emotion Trends Over Time for {selected_name}',
                                    width=600,
                                    height=400
                                ).interactive()
                                
                                st.altair_chart(admin_time_chart, use_container_width=True)
                            else:
                                st.write("No text emotion data available for selected user.")
                        
                        with admin_tab2:
                            if all_image_emotion_data:
                                df_admin_img_time = pd.DataFrame(all_image_emotion_data, columns=["Username", "Emotion", "Timestamp"])
                                df_admin_img_time['Timestamp'] = pd.to_datetime(df_admin_img_time['Timestamp'])
                                df_admin_img_time['Date'] = df_admin_img_time['Timestamp'].dt.date
                                
                                # Group by day and emotion to count occurrences
                                admin_img_emotion_counts = df_admin_img_time.groupby(['Date', 'Emotion']).size().reset_index(name='Count')
                                
                                # Create time-series chart
                                admin_img_time_chart = alt.Chart(admin_img_emotion_counts).mark_line(point=True).encode(
                                    x='Date:T',
                                    y='Count:Q',
                                    color='Emotion:N',
                                    tooltip=['Date', 'Emotion', 'Count']
                                ).properties(
                                    title=f'Image Emotion Trends Over Time for {selected_name}',
                                    width=600,
                                    height=400
                                ).interactive()
                                
                                st.altair_chart(admin_img_time_chart, use_container_width=True)
                            else:
                                st.write("No image emotion data available for selected user.")
                    else:
                        st.write("No emotion trend data available for the selected user.")

    # else:
    #     st.warning("Please login or register to access the app.")

if __name__ == '__main__':
    main()
