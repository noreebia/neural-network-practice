from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
    if column == "tnp" or column == "twp" or column == "iap" or column == "fmi" or column == "nf" or column == "sh" or column == "atd":
        x[column] = labelEncoder.fit_transform(x[column])


columnsToEncode = ["ge", "cst", "arr", "ms", "ls", "as", "fo", "mo", "me", "fs", "tt", "fq", "mq", "ss"]
oneHotEncodedX = pd.get_dummies(x, columns = columnsToEncode)

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
  
accuracy = svm_model_linear.score(xTest, yTest) 
print(accuracy)

###

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(xTrain, yTrain) 
  
accuracy = knn.score(xTest, yTest) 
print(accuracy)

###
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)
accuracy = clf.score(xTest, yTest)
print(accuracy)



