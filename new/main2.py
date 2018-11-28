from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# purely one hot encoded

data = arff.loadarff('data.arff')
totalData = pd.DataFrame(data[0])
espColumn = totalData['esp']
x = totalData.drop(columns = "esp")

oneHotEncodedX = pd.get_dummies(x)
oneHotEncodedX = oneHotEncodedX.values

labelEncoder = LabelEncoder()
labelEncodedY = labelEncoder.fit_transform(espColumn)

print(oneHotEncodedX)
print(labelEncodedY)
print("{} , {}".format(oneHotEncodedX.shape, labelEncodedY.shape))

xTrain, xTest, yTrain, yTest = train_test_split(oneHotEncodedX, labelEncodedY, test_size = 0.3)

print("{} , {}".format(xTrain.shape, yTrain.shape))
print("{} , {}".format(xTest.shape, yTest.shape))

###

from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(xTrain, yTrain) 
svm_predictions = svm_model_linear.predict(xTest) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(xTest, yTest) 
print(accuracy)

###

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(xTrain, yTrain) 
  
# accuracy on X_test 
accuracy = knn.score(xTest, yTest) 
print(accuracy)
