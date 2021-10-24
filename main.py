import os
import signal
import tkinter as tk
import PIL
from PIL import Image, ImageTk
from tkinter import *
import subprocess
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)


root = tk.Tk()
root.title("Speech Emotion Recognition")
canvas = tk.Canvas(root)
canvas.grid(columnspan=1, rowspan=4)

# logo
logo = PIL.Image.open("Img/logo.png")
logo = logo.resize((250, 250))
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(column=0, row=0, padx=5, pady=50)

# Emotion
label = tk.Label(root, text="None", font="Raleway")
label.grid(column=0, row=1)


pid = None


def start():
    btn["command"] = stop
    btn["text"] = "Stop"
    btn["bg"] = "red"

    global pid
    p = subprocess.Popen(["python3", "rec.py"])
    pid = p.pid
    #p.wait()


def stop():
    btn["command"] = start
    btn["text"] = "Record"
    btn["bg"] = "#20bebe"

    global pid
    if pid is not None:
        os.kill(pid, signal.SIGINT)
        pid = None


# button
btn = tk.Button(root, text="Record", command=lambda:start(), font="Raleway", bg="#20bebe", fg="white", height=2, width=15)
btn.grid(column=0, row=2, pady=50)







def cng_emotion(emo):
    if emo == "neutral":
        logo = PIL.Image.open("Img/neutral.png")
    elif emo == "happy":
        logo = PIL.Image.open("Img/happy.png")
    elif emo == "sad":
        logo = PIL.Image.open("Img/sad.png")
    elif emo == "angry":
        logo = PIL.Image.open("Img/angry.png")

    logo = logo.resize((250, 250))
    logo = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(image=logo)
    logo_label.image = logo
    logo_label.grid(column=0, row=0, padx=5, pady=50)

# def test(label, btn_text):
#     # btn_text.set("Loading..!")
#     emotion = ts.data()
#     self.cng_emotion(emotion)
#     label.config(text=emotion)
root.mainloop()