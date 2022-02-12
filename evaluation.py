import pickle
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict

model = pickle.load(open("result/Emotion/mlp_classifier.model", "rb"))
X_train = pickle.load(open("result/Emotion/X_train.variable", "rb"))
X_test = pickle.load(open("result/Emotion/X_test.variable", "rb"))
y_train = pickle.load(open("result/Emotion/y_train.variable", "rb"))
y_test = pickle.load(open("result/Emotion/y_test.variable", "rb"))
y_pred = pickle.load(open("result/Emotion/y_pred.variable", "rb"))


classification_report = classification_report(y_test, y_pred);
print(classification_report);

confusion_mat = confusion_matrix(y_test, y_pred)
sb.heatmap(confusion_mat, annot=True)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Metrix")
plt.show()


#10-fold cross validation
from sklearn.preprocessing import LabelEncoder
l_encode = LabelEncoder()
y_train = l_encode.fit_transform(y_train)
kfold = cross_val_score(model, X_train, y_train, scoring='r2', cv=10)

print("K-fold array:")
print(kfold)

print("K-fold mean:")
print(np.mean(kfold))

l_encode = LabelEncoder()
y_test = l_encode.fit_transform(y_test)
pred = cross_val_predict(model, X_test, y_test)
score_test = cross_val_score(model, X_test, y_test, cv=10)
print("Score Means:")
print(np.mean(score_test))
