# ------------------------------------------------------------------
# Build and Evaluate the Logistic Regression Model
# ------------------------------------------------------------------

# Import Libraries
import pandas as pd

# Read the data and Create a copy
LoanData = pd.read_csv("01Exercise1.csv")
LoanPrep = LoanData.copy()


#find out columns with missing values
LoanPrep.isnull().sum(axis=0)


# Replace Missing Values. Drop the rows.
LoanPrep = LoanPrep.dropna()

# Drop irrelevant columns based on business sense
LoanPrep = LoanPrep.drop(['gender'], axis=1)

# Create Dummy variables
LoanPrep.dtypes
LoanPrep = pd.get_dummies(LoanPrep, drop_first=True)


# Normalize the data (Income and Loan Amount) Using StandardScaler
from sklearn.preprocessing import StandardScaler
scalar_ = StandardScaler()

LoanPrep['income'] = scalar_.fit_transform(LoanPrep[['income']])
LoanPrep['loanamt'] = scalar_.fit_transform(LoanPrep[['loanamt']])


# Create the X (Independent) and Y (Dependent) dataframes
# -------------------------------------------------------
Y = LoanPrep[['status_Y']]
X = LoanPrep.drop(['status_Y'], axis=1)


# Split the X and Y dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)


# Build the Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)

# Predict the outcome using Test data
Y_predict = lr.predict(X_test)

# import libraries to evaluate the model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Build Confusion Matrix, score and report for the default model
cm1 = confusion_matrix(Y_test, Y_predict)
score1 = lr.score(X_test, Y_test)
cr1 = classification_report(Y_test, Y_predict)

# Create prediction probability list
Y_prob = lr.predict_proba(X_test)[:, 1]

# Create new predictions based on new probability threshold
Y_new_pred = []
threshold  = 0.8

for i in range(0, len(Y_prob)):
    if Y_prob[i] > threshold:
        Y_new_pred.append(1)
    else:
        Y_new_pred.append(0)
        
# Check the effect of probability threshold on predictions
cm2 = confusion_matrix(Y_test, Y_new_pred)
score2 = accuracy_score(Y_test, Y_new_pred)
cr2 = classification_report(Y_test, Y_new_pred)

# Understand and implement AUC ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

# Get the Area Under the ROC Curve
auc = roc_auc_score(Y_test, Y_prob)

# plot ROC
import matplotlib.pyplot as plt

fpr, tpr, threshold = roc_curve(Y_test, Y_prob)
plt.plot(fpr, tpr, linewidth=4)
plt.xlabel("False Positive rate")
plt.ylabel("True Positve Rate")
plt.title("ROC Curve for Loan Prediction")
plt.grid()













