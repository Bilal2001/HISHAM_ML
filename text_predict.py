

import random
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))


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

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮", "no face": "😐"
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results



raw_text = input("Enter the raw_text: ")
prediction = predict_emotions(raw_text)
probability = get_prediction_proba(raw_text)


# print(f"Prediction: {prediction}, Probability: {probability}")
emoji_icon = emotions_emoji_dict[prediction]
recommendation = recommendation_text[prediction]
print("{}: {}".format(prediction.capitalize(), emoji_icon))
print("Confidence: {}".format(np.max(probability)))
print("Suggestion: {}".format(recommendation[random.randint(0, len(recommendation)-1)]))
