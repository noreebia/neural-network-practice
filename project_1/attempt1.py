from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# combination of one-hot-encoding and label-encoding

data = arff.loadarff('data.arff')
totalData = pd.DataFrame(data[0])
espColumn = totalData['esp']
x = totalData.drop(columns = "esp")

# Label encode x
labelEncoder = LabelEncoder()
for column in x:
    x[column] = x[column].str.decode("utf-8")
    if column == "tnp" or column == "twp" or column == "iap" or column == "fmi" or column == "nf" or column == "sh" or column == "atd":
        x[column] = labelEncoder.fit_transform(x[column])

# One-hot encode x
columnsToEncode = ["ge", "cst", "arr", "ms", "ls", "as", "fo", "mo", "me", "fs", "tt", "fq", "mq", "ss"]
x = pd.get_dummies(x, columns = columnsToEncode).values

# Label encode dependent variable to make y
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(espColumn)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
print("Integer encoding combined with one-hot encoding\n")
print("Gaussian Naive Bayes\n", cross_val_score(GaussianNB(), x, y, cv=5, scoring='accuracy').mean())
print("\nSVM\n", cross_val_score(SVC(kernel = 'linear', C = 1), x, y, cv=kf, scoring='accuracy').mean())
print("\nKNN\n", cross_val_score(KNeighborsClassifier(n_neighbors = 7), x, y, cv=kf, scoring='accuracy').mean())
print("\nRandom Forest(n_jobs=2)\n", cross_val_score(RandomForestClassifier(n_jobs=2, random_state=0), x, y, cv=kf, scoring='accuracy').mean())
print("\nRandom Forest(10 estimators)\n", cross_val_score(RandomForestClassifier(n_estimators = 10), x, y, cv=kf, scoring='accuracy').mean())

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver="lbfgs", max_iter=500)
print("\nMLP Classifier\n", cross_val_score(classifier, x, y, cv=kf, scoring='accuracy').mean())
