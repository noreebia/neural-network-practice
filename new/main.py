from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
data = arff.loadarff('data.arff')
totalData = pd.DataFrame(data[0])
espColumn = totalData['esp']
x = totalData.drop(columns = "esp")

# combination of one-hot-encoding and label-encoding

labelEncoder = LabelEncoder()

for column in x:
    x[column] = x[column].str.decode("utf-8")
    if column == "tnp" or column == "twp" or column == "iap" or column == "fmi" or column == "fs" or column == "fq" or column == "mq" or column == "nf" or column == "sh" or column == "ss" or column == "tt" or column == "atd":
        x[column] = labelEncoder.fit_transform(x[column])

print(x.to_string())

columnsToEncode = ["ge", "cst", "arr", "ms", "ls", "as", "fo", "mo", "ss", "me"]
oneHotEncodedX = pd.get_dummies(x, columns = columnsToEncode)

print(oneHotEncodedX.to_string())
oneHotEncodedX = oneHotEncodedX.values

labelEncoder = LabelEncoder()
labelEncodedY = labelEncoder.fit_transform(espColumn)

print(oneHotEncodedX)
print(labelEncodedY)
print("{} , {}".format(oneHotEncodedX.shape, labelEncodedY.shape))

xTrain, xTest, yTrain, yTest = train_test_split(oneHotEncodedX, labelEncodedY, test_size = 0.3)

print("{} , {}".format(xTrain.shape, yTrain.shape))
print("{} , {}".format(xTest.shape, yTest.shape))

# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(xTrain, yTrain) 
dtree_predictions = dtree_model.predict(xTest) 
  
# creating a confusion matrix 
cm = confusion_matrix(yTest, dtree_predictions) 
print(cm)

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
