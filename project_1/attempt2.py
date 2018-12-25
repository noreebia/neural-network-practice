from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = arff.loadarff('data.arff')
totalData = pd.DataFrame(data[0])
espColumn = totalData['esp']
independentVariables = totalData.drop(columns = "esp")

# one hot encode x
x = pd.get_dummies(independentVariables).values

labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(espColumn)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
print("Pure one-hot encoding\n")
print("SVM\n",cross_val_score(SVC(kernel = 'linear', C = 1), x, y, cv=kf, scoring='accuracy').mean())
print("\nKNN\n", cross_val_score(KNeighborsClassifier(n_neighbors = 7), x, y, cv=kf, scoring='accuracy').mean())
print("\nRandom Forest(n_jobs=2)\n", cross_val_score(RandomForestClassifier(n_jobs=2), x, y, cv=kf, scoring='accuracy').mean())
print("\nRandom Forest(n_estimators=10)\n", cross_val_score(RandomForestClassifier(n_estimators = 10), x, y, cv=kf, scoring='accuracy').mean())

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver="lbfgs", max_iter=1000)
print("\nMLP Classifier\n", cross_val_score(classifier, x, y, cv=kf, scoring='accuracy').mean())