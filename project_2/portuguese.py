from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

independentVariables = pd.read_csv('student-por.csv', sep=";")

dependentVariables = independentVariables["G3"]

independentVariables = independentVariables.drop(columns = "G3")

categoricalFeatures = []
columns = list(independentVariables.columns.values)
for column in columns:
    if independentVariables[column].dtype != "int64":
        categoricalFeatures.append(column)

independentVariables = pd.get_dummies(independentVariables, columns = categoricalFeatures)
dependentVariables = dependentVariables.to_frame()
dependentVariables["G3"] = pd.cut(dependentVariables["G3"], [-1,4,8,12,20])

labelEncoder = LabelEncoder()
dependentVariables["G3"] = labelEncoder.fit_transform(dependentVariables["G3"])

x = independentVariables.values
y = dependentVariables.values.ravel()

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

kf = KFold(n_splits=10, shuffle=True)
print("Portuguese Grade Estimation\n")
print("SVM\n", cross_val_score(SVC(kernel = 'linear', C = 1), x, y, cv=kf, scoring='accuracy').mean())
print("\nKNN\n", cross_val_score(KNeighborsClassifier(n_neighbors = 7), x, y, cv=kf, scoring='accuracy').mean())
print("\nRandom Forest\n", cross_val_score(RandomForestClassifier(), x, y, cv=kf, scoring='accuracy').mean())
print("\nRandom Forest(n_jobs=2)\n", cross_val_score(RandomForestClassifier(n_jobs=2), x, y, cv=kf, scoring='accuracy').mean())
print("\nRandom Forest(n_estimators=10)\n", cross_val_score(RandomForestClassifier(n_estimators = 10), x, y, cv=kf, scoring='accuracy').mean())

classifier = MLPClassifier(solver="lbfgs")
print("\nMLP classifier\n", cross_val_score(classifier, x, y, cv=kf, scoring='accuracy').mean())