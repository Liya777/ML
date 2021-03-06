import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from google.colab import files
files.upload()

ecg = pd.read_csv("data.csv",encoding = "ISO-8859-1")

ecg.shape

ecg.head()

feature_cols = ['0.12668','0.16466',	'0.40578',	'0.66713',	'0.59824',	'0.032789',	'0.22018',	'0.1124',	'0.11888',	'0.34479',	'0.47566',	'0.27085',	'0.021959',	'0.18626',	'0.10393',	'0.31167',	'0.55827','0.28279',	'0.13941',	'0.050422',	'0.31311']
X = ecg[feature_cols] 
label = ['0']
y = ecg[label]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear', gamma=1)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
print("TP = "+str(TP))
print("FP = "+str(FP))
print("FN = "+str(FN))
print("TN = "+str(TN))
precision = np.sum(TP / (TP + FP))
recall = np.sum(TP / (TP + FN))
F1 = 2 * (precision * recall) / (precision + recall)
print ("precision= ",precision)
print("recall",recall)
print("F1= ",F1)
prediction=svclassifier.predict(X_test)
print("confusion matrix:\n",metrics.confusion_matrix(prediction,y_test))
