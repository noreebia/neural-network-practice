from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#np.set_printoptions(threshold=np.inf)

independentVariables = pd.read_csv('student-mat.csv', sep=";")
# pd.concat([data[[0]], data[1].str.split(';', expand=True)], axis=1)

dependentVariables = independentVariables["G3"]

independentVariables = independentVariables.drop(columns = "G3")

print(independentVariables.dtypes)


categoricalFeatures = []
columns = list(independentVariables.columns.values)
for column in columns:
    if independentVariables[column].dtype != "int64":
        categoricalFeatures.append(column)
print(categoricalFeatures)

independentVariables = pd.get_dummies(independentVariables, columns = categoricalFeatures)
print(independentVariables)

print(dependentVariables)

print(dependentVariables.max())
print(dependentVariables.min())

dependentVariables = dependentVariables.to_frame()

dependentVariables["G3"] = pd.cut(dependentVariables["G3"], [-1,4,8,12,20])
print(dependentVariables)

labelEncoder = LabelEncoder()
dependentVariables["G3"] = labelEncoder.fit_transform(dependentVariables["G3"])

print(independentVariables.values)
print(independentVariables.values.shape)

print(dependentVariables.values.ravel())
print(dependentVariables.values.ravel().shape)

x = independentVariables.values
y = dependentVariables.values.ravel()


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3)

print("{} , {}".format(xTrain.shape, yTrain.shape))
print("{} , {}".format(xTest.shape, yTest.shape))


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, random_state = 42)
classifier.fit(xTrain, yTrain)

predictions = classifier.predict(xTest)
accuracy = classifier.score(xTest, yTest)
print(accuracy)


