from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

data = arff.loadarff('data.arff')
totalData = pd.DataFrame(data[0])
espColumn = totalData['esp']
x = totalData.drop(columns = "esp")

oneHotEncodedX = pd.get_dummies(x)
oneHotEncodedX = oneHotEncodedX.values

labelEncoder = LabelEncoder()
labelEncodedY = labelEncoder.fit_transform(espColumn)
labelEncodedY = labelEncodedY.reshape(labelEncodedY.shape[0], -1)

print(oneHotEncodedX)
print(labelEncodedY)
print("{} , {}".format(oneHotEncodedX.shape, labelEncodedY.shape))