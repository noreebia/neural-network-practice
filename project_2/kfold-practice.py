from sklearn.model_selection import KFold # import KFold
from sklearn.cross_validation import cross_val_score
import numpy as np
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array
y = np.array([1, 2, 3, 4]) # Create another array
kf = KFold(n_splits=2) # Define the split - into 2 folds 
asdf = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
KFold(n_splits=2, random_state=None, shuffle=False)

for train_index, test_index in kf.split(X):
    print("Train index: {} , Test index: {}".format(train_index, test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("X_train: {}\nX_test: {}".format(X_train, X_test))
    print("y_train: {}\ny_test: {}".format(y_train, y_test))
