
import sys
import numpy as np
import sklearn as sk
from sklearn import tree
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Use: python " + __file__ + " train test")
        print("train = train file.")
        print("test = test file.")
        quit()

    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]

    X_train, y_train = load_svmlight_file(train_file_name)
    X_test, y_test = load_svmlight_file(test_file_name)
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Accuracy: " + str(round(score * 100, 4)) + " %")
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    m = metrics.confusion_matrix(y_test, pred)
    print("Tested: " + str(len(y_test)))
    print(m)
