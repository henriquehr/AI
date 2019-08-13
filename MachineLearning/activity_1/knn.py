#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import pylab as pl
from sklearn.metrics import classification_report


def main(data):

        # loads data
        print "Loading data..."
        X_data, y_data = load_svmlight_file(data)
        # splits data
        print "Spliting data..."
        X_train, X_test, y_train, y_test =  cross_validation.train_test_split(X_data, y_data, test_size=0.3)

        # fazer a normalizacao dos dados #######
        #scaler = preprocessing.MinMaxScaler()
        #X_train_minmax = scaler.fit_transform(X_train)
        #X_test_minmax = scaler.transform(X_test)

        # cria um kNN
        neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1)

        print 'Fitting knn'
        neigh.fit(X_train, y_train)

        # predicao do classificador
        print 'Predicting...'
        y_pred = neigh.predict(X_test)

        # mostra o resultado do classificador na base de teste
        print neigh.score(X_test, y_test)

        # cria a matriz de confusao
        cm = confusion_matrix(y_test, y_pred)
        print cm
	#print classification_report(y_test, y_pred)
	#pl.matshow(cm)
	#pl.colorbar()
	#pl.show()

if __name__ == "__main__":
        if len(sys.argv) != 2:
                sys.exit("Use: knn.py <data>")

        main(sys.argv[1])


