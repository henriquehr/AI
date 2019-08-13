import numpy as np
import matplotlib.pyplot as plt
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


print ("Loading Data")
data = pandas.read_csv("usina72.csv")

feature_cols = ['f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12']
X = data.loc[:,feature_cols]
label_col = ['f3']
Y = data.loc[:,label_col]

## split the data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.5, random_state=0)

### Normalizacao
#scaler = preprocessing.MinMaxScaler()
#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)

errors = []
errors_norm = []
labels = ['Neural Network', 'SVR', 'KNN', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Linear']

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
X_test_orig = X_test

for i in range(0, 2):
    errors_tmp = []
    if i == 1:
        print("Normalizing...")
        scaler = preprocessing.MinMaxScaler()
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    print("Neural Network...")
    model = Sequential()
    model.add(Dense(256, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adadelta') #adam
    model.fit(X_train, y_train, epochs=100, validation_split=0.33, verbose=0)
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    errors_tmp.append(MSE)
    print("   MSE: " + str(MSE))

    print("SVR...")
    clf = SVR(C=1.0, epsilon=0.1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    MSE = mean_squared_error(y_test, pred)
    errors_tmp.append(MSE)
    print("   MSE: " + str(MSE))

    print("KNN...")
    clf = KNeighborsRegressor(n_neighbors=5)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    MSE = mean_squared_error(y_test, pred)
    errors_tmp.append(MSE)
    print("   MSE: " + str(MSE))

    print("Decison Tree...")
    clf = DecisionTreeRegressor(criterion='mse')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    MSE = mean_squared_error(y_test, pred)
    errors_tmp.append(MSE)
    print("   MSE: " + str(MSE))

    print("Random Forest...")
    clf = RandomForestRegressor(criterion='mse', n_estimators=100)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    MSE = mean_squared_error(y_test, pred)
    errors_tmp.append(MSE)
    print("   MSE: " + str(MSE))

    print("Gradient Boosting...")
    clf = GradientBoostingRegressor(criterion='mse', n_estimators=100)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    MSE = mean_squared_error(y_test, pred)
    pred_svr = pred
    errors_tmp.append(MSE)
    print("   MSE: " + str(MSE))

    print("Linear...")
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    MSE = mean_squared_error(y_test, pred)
    errors_tmp.append(MSE)
    print("   MSE: " + str(MSE))
    if i == 1:
        errors_norm = errors_tmp
    else: 
        errors = errors_tmp

bar_width = 0.3
index = np.arange(len(labels))
_, ax = plt.subplots()
ax.bar(index, errors, bar_width, label='Without normalization')
ax.bar(index + bar_width, errors_norm, bar_width, label='With normalization')
#ax.set_xlabel("Classifiers")
ax.set_ylabel("MSE")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(labels)
plt.legend()
plt.show()
