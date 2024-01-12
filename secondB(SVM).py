import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv("heart_disease.csv")

# Separate the data
x = data.drop('target', axis=1)
y = data.target

# Split the test set and train set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=109)

# Kernels to compare
kernels = ['rbf', 'poly', 'sigmoid']

# Iterate over different kernels
for kernel in kernels:
    # Create an SVM Classifier with the specified kernel
    ml = svm.SVC(kernel=kernel)

    # Train the model using the training sets
    ml.fit(x_train, y_train)

    # Predict the response for the test dataset
    y_pred = ml.predict(x_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f'\nConfusion Matrix for {kernel} kernel:')
    print(cm)

    # Performance metrics
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print('False Positives\n {}'.format(FP))
    print('False Negatives\n {}'.format(FN))
    print('True Positives\n {}'.format(TP))
    print('True Negatives\n {}'.format(TN))
    TPR = TP / (TP + FN)
    print('Sensitivity \n {}'.format(TPR))
    TNR = TN / (TN + FP)
    print('Specificity \n {}'.format(TNR))
    Precision = TP / (TP + FP)
    print('Precision \n {}'.format(Precision))
    Recall = TP / (TP + FN)
    print('Recall \n {}'.format(Recall))
    Acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy \n{}'.format(Acc))
    Fscore = 2 * (Precision * Recall) / (Precision + Recall)
    print('FScore \n{}'.format(Fscore))
    print('\n---------------------------------------')
