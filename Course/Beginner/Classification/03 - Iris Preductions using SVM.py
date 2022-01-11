# ---------------------------------------------------------------
# IRIS Plant Classification
# ---------------------------------------------------------------

# import and load the Iris Dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target


# split, train test....
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)

# Train the SVC 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Gamma as 1.0
svc = SVC(kernel='rbf', gamma=1.0)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)

cm_rbf01 = confusion_matrix(Y_test, Y_predict)

# Gamma as 10
svc = SVC(kernel='rbf', gamma=10)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)

cm_rbf10 = confusion_matrix(Y_test, Y_predict)


# Linear Kernel
svc = SVC(kernel='linear')
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
cm_linear = confusion_matrix(Y_test, Y_predict)


# Polynomial Kernel
svc = SVC(kernel='poly')
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
cm_poly = confusion_matrix(Y_test, Y_predict)


# Sigmoid Kernel
svc = SVC(kernel='sigmoid')
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
cm_sig = confusion_matrix(Y_test, Y_predict)








