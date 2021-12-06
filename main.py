import os
import pickle
import signal
import tkinter as tk
import PIL
from PIL import Image, ImageTk
from tkinter import *
from tkinter import filedialog
import subprocess
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)
from helper import *
from playsound import playsound
from pydub import AudioSegment, effects

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
gender_label.grid(column=0, row=2)


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
    if gend == "M":
        gender_label.config(text="Male")
    elif gend == "F":
        gender_label.config(text="Female")
    else:
        gender_label.config(text="Undefined")


def gender_test(filename):
    # model_for_gender = pickle.load(open("result/Gender/mlp_classifier.model", "rb"))
    # # extract features and reshape it
    # features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # # predict
    # result = model_for_gender.predict(features)[0]

    model_for_gender = pickle.load(open("result/Gender/mlp_classifier(ds_file).model", "rb"))
    # extract features and reshape it
    features = None
    # frequency = get_frequencies(filename)
    # if len(frequency) >= 0:
    #     nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr = get_features(frequency)
    #     # predict
    #     result = model_for_gender.predict([[nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr]])[0]
    # else: result = "Undefined"

    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = model_for_gender.predict(features)[0]

    # show the result !
    print("result:", result)
    cng_gender(result)

pid = None


def start():
    btn["command"] = stop
    btn["image"] = btn_logo_stop

    fl = "Recordings/audio.wav"
    if fl:
        os.remove(fl)
    global pid
    p = subprocess.Popen(["python3", "rec.py"])
    pid = p.pid
    #p.wait()


def stop():
    btn["command"] = start
    btn["image"] = btn_logo_rec

    global pid
    if pid is not None:
        os.kill(pid, signal.SIGINT)
        pid = None
    src = "Recordings/rec.wav"
    filename = "Recordings/audio.wav"
    if src and filename:
        os.system(f"ffmpeg -i {src} -ac 1 -ar 16000 {filename}")
    if src:
        os.remove(src)

    # ##Normalize
    # rawsound = AudioSegment.from_file(filename, "wav")
    # filename = effects.normalize(rawsound)
    # filename.export(filename, format="wav")
    playsound(filename)
    emotion_test(filename)
    gender_test(filename)


def browseFiles():
    filename = filedialog.askopenfilename(initialdir="/home/siasibul/PycharmProjects/Human-Speech-Emotion-Recognition-System/Recordings/test/", title="Select a File", filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
    playsound(filename)
    emotion_test(filename)
    gender_test(filename)


# button
btn_logo_rec = PIL.Image.open("Img/btn_record.png")
btn_logo_rec = btn_logo_rec.resize((50, 50))
btn_logo_rec = ImageTk.PhotoImage(btn_logo_rec)

btn_logo_stop = PIL.Image.open("Img/btn_stop.png")
btn_logo_stop = btn_logo_stop.resize((50, 50))
btn_logo_stop = ImageTk.PhotoImage(btn_logo_stop)

btn_upload_logo = PIL.Image.open("Img/btn_upload.png")
btn_upload_logo = btn_upload_logo.resize((50, 50))
btn_upload_logo = ImageTk.PhotoImage(btn_upload_logo)


btn_frame = tk.Frame(root)
btn_frame.grid(column=0, row=3, columnspan=2, rowspan=1, padx=50, pady=50)
btn = tk.Button(btn_frame, image=btn_logo_rec, borderwidth=0, border=0, command=lambda:start())
btn.grid(column=0, row=0, padx=10, pady=10)
file_btn = tk.Button(btn_frame, image=btn_upload_logo, borderwidth=0, border=0, command=browseFiles)
file_btn.grid(column=1, row=0, padx=10, pady=10)


root.mainloop()