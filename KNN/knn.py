
from google.colab import files
files.upload()

import pandas as pd
import numpy as np

data = pd.read_csv("final.csv",encoding = "ISO-8859-1")

data.head()

data=data.drop(['Id',   'Company','Educational details',    'jobdescription',   'jobid',    'numberofpositions',    'payrate'], axis=1)

feature_cols = ['Education', 'Experience', 'jobtitle', 'loc_1']
X = data[feature_cols] 
label = ['industry']
y = data[label]

data['loc_1']=pd.factorize(data.loc_1)[0]
data['industry']=pd.factorize(data.industry)[0]
data['Education']=pd.factorize(data.Education)[0]
data['jobtitle']=pd.factorize(data.jobtitle)[0]

data.head()

X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
