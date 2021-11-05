import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("gender.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# l_encode = LabelEncoder()
# y = l_encode.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

model = MLPClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred_vertical = y_pred.reshape(len(y_pred), 1)

y_true_vertical = y_test.reshape(len(y_test), 1)

true_pred = np.concatenate((y_true_vertical, y_pred_vertical), axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/Gender/mlp_classifier(gender.csv).model", "wb"))