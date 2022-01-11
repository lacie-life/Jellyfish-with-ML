# -----------------------------------------------------------------
# Compare multiple Classifiers for different train and test values
# -----------------------------------------------------------------

# Import libraries
import pandas as pd

# Read dataset
data = pd.read_csv('04 - decisiontreeAdultIncome.csv')

# Create Dummy variables
data_prep = pd.get_dummies(data, drop_first=True)

# Create X and Y Variables
X = data_prep.iloc[:, :-1]
Y = data_prep.iloc[:, -1]


# Import and train Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1234)

# Import and train Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)

# Import and train Support Vector Classifier
from sklearn.svm import SVC
svc = SVC(kernel='rbf', gamma=0.5)

# Import and perform cross validation
from sklearn.model_selection import cross_validate
cv_results_dtc = cross_validate(dtc, X, Y, cv=10, return_train_score=True)
cv_results_rfc = cross_validate(rfc, X, Y, cv=10, return_train_score=True)
cv_results_svc = cross_validate(svc, X, Y, cv=10, return_train_score=True)

# Get the average of all the results
import numpy as np
dtc_test_average = np.average(cv_results_dtc['test_score'])
rfc_test_average = np.average(cv_results_rfc['test_score'])
svc_test_average = np.average(cv_results_svc['test_score'])

dtc_train_average = np.average(cv_results_dtc['train_score'])
rfc_train_average = np.average(cv_results_rfc['train_score'])
svc_train_average = np.average(cv_results_svc['train_score'])

# print the results 
print()
print()
print('        ','Decision Tree  ', 'Random Forest  ','Support Vector   ')
print('        ','---------------', '---------------','-----------------')

print('Test  : ',
      round(dtc_test_average, 4), '        ',
      round(rfc_test_average, 4), '        ',
      round(svc_test_average, 4))

print('Train : ',
      round(dtc_train_average, 4), '        ',
      round(rfc_train_average, 4), '        ',
      round(svc_train_average, 4))




