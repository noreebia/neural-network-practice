from scipy.io import arff
import pandas as pd

data = arff.loadarff('data.arff')
df = pd.DataFrame(data[0])

espColumn = df['esp']

df = df.drop(columns = "esp")

#print(df)
#print(espColumn)
#print(pd.get_dummies(df))
#print(df)

oneHotEncodedX = pd.get_dummies(df)
oneHotEncodedY = pd.get_dummies(espColumn)

print(oneHotEncodedX)
print(oneHotEncodedY)

print(espColumn)
