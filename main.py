import os
import pickle
import signal
import tkinter as tk
import PIL
from PIL import Image, ImageTk
from tkinter import *
import subprocess
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)
from helper import *

root = tk.Tk()
root.title("Speech Emotion Recognition")
canvas = tk.Canvas(root)
canvas.grid(columnspan=1, rowspan=4)

# emotion_icon
emotion_icon = PIL.Image.open("Img/logo.png")
emotion_icon = emotion_icon.resize((250, 250))
emotion_icon = ImageTk.PhotoImage(emotion_icon)
emotion_icon_label = tk.Label(image=emotion_icon)
emotion_icon_label.image = emotion_icon
emotion_icon_label.grid(column=0, row=0, padx=5, pady=50)

# Emotion
emotion_label = tk.Label(root, text="Emotion", font="Raleway")
emotion_label.grid(column=0, row=1)

# Gender
gender_label = tk.Label(root, text="Gender", font="Raleway")
gender_label.grid(column="0", row="2")

pid = None


def start():
    btn["command"] = stop
    btn["text"] = "Stop"
    btn["bg"] = "red"

    fl = "Recordings/audio.wav"
    os.remove(fl)
    global pid
    p = subprocess.Popen(["python3", "rec.py"])
    pid = p.pid
    #p.wait()


def cng_emotion(emo):
    global emotion_icon
    emotion_icon = PIL.Image.open("Img/neutral.png")
    if emo == "NEUTRAL":
        emotion_icon = PIL.Image.open("Img/neutral.png")
        emotion_label.config(text="Neutral")
    elif emo == "HAPPY":
        emotion_icon = PIL.Image.open("Img/happy.png")
        emotion_label.config(text="Happy")
    elif emo == "SAD":
        emotion_icon = PIL.Image.open("Img/sad.png")
        emotion_label.config(text="Sad")
    elif emo == "ANGRY":
        emotion_icon = PIL.Image.open("Img/angry.png")
        emotion_label.config(text="Angry")
    elif emo == "DISGUST":
        emotion_icon = PIL.Image.open("Img/disgused.png")
        emotion_label.config(text="Disgust")
    elif emo == "FEAR":
        emotion_icon = PIL.Image.open("Img/fear.png")
        emotion_label.config(text="Fear")
    elif emo == "SURPRISE":
        emotion_icon = PIL.Image.open("Img/surprised.png")
        emotion_label.config(text="Surprised")

    emotion_icon = emotion_icon.resize((250, 250))
    emotion_icon = ImageTk.PhotoImage(emotion_icon)
    emotion_icon_label = tk.Label(image=emotion_icon)
    emotion_icon_label.image = emotion_icon
    emotion_icon_label.grid(column=0, row=0, padx=5, pady=50)


def emotion_test(filename):
    model_for_emotion = pickle.load(open("result/Emotion/mlp_classifier.model", "rb"))
    # extract features and reshape it
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = model_for_emotion.predict(features)[0]
    # show the result !
    print("result:", result)
    cng_emotion(result)


def cng_gender(gend):
    if gend == "Male":
        gender_label.config(text="Male")
    elif gend == "Female":
        gender_label.config(text="Female")
    else:
        gender_label.config(text="Undefined")


def gender_test(filename):
    # model_for_gender = pickle.load(open("result/Gender/mlp_classifier.model", "rb"))
    # # extract features and reshape it
    # features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # # predict
    # result = model_for_gender.predict(features)[0]

    model_for_gender = pickle.load(open("result/Gender/mlp_classifier_1.model", "rb"))
    # extract features and reshape it
    features = None
    frequency = get_frequencies(filename)
    if len(frequency) >= 0:
        nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr = get_features(frequency)
        # predict
        result = model_for_gender.predict([[nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr]])[0]
    else: result = "Undefined"
    # show the result !
    print("result:", result)
    cng_gender(result)

def stop():
    btn["command"] = start
    btn["text"] = "Record"
    btn["bg"] = "#20bebe"

    global pid
    if pid is not None:
        os.kill(pid, signal.SIGINT)
        pid = None
    src = "Recordings/rec.wav"
    filename = "Recordings/audio.wav"
    os.system(f"ffmpeg -i {src} -ac 1 -ar 16000 {filename}")
    os.remove(src)
    filename = "Recordings/3.wav"
    emotion_test(filename)
    gender_test(filename)

# button
btn = tk.Button(root, text="Record", command=lambda:start(), font="Raleway", bg="#20bebe", fg="white", height=2, width=15)
btn.grid(column=0, row=3, pady=50)


# def test(label, btn_text):
#     # btn_text.set("Loading..!")
#     emotion = ts.data()
#     self.cng_emotion(emotion)
#     label.config(text=emotion)

root.mainloop()