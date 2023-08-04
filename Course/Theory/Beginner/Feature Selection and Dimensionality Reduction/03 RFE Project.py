# -----------------------------------------------------------------
# Implement Recursive Feature Elimination.
# Predict product purchase for the Bank Telemarketing dataset
# -----------------------------------------------------------------

# Import libraries
import pandas as pd

# Read the file
f = pd.read_csv('bank.csv')
f = f.drop("duration", axis = 1)

# Split the columns into Dependent (Y) and independent (X) features
x = f.iloc[:,:-1]
y = f.iloc[:, -1]


# Create dummy variables
x = pd.get_dummies(x, drop_first=True)
y = pd.get_dummies(y, drop_first=True)


# Split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(x, y, test_size = 0.3, random_state = 1234, stratify=y)

# Import Randon Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

# Default Random Forest Object
rfc1 = RandomForestClassifier(random_state=1234)
rfc1.fit(X_train, Y_train)
Y_predict1 = rfc1.predict(X_test)


# Score and Evaluate the model 
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_test, Y_predict1)
score1 = rfc1.score(X_test, Y_test)


# Apply Recursive Feature Elimination
from sklearn.feature_selection import RFE
rfc2 = RandomForestClassifier(random_state=1234)

# Create an RFE selector object using RFC as an estimator
rfe = RFE(estimator=rfc2, n_features_to_select=30, step=1)

# Fit the data to the rfe selector
rfe.fit(x, y)

# Create new Train and Test datasets
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Fit the Random Forest classifier to the new train and test with 80 features
rfc2.fit(X_train_rfe, Y_train)

# Test the model with new Test dataset
Y_predict = rfc2.predict(X_test_rfe)

# Score and Evaluate the new model 
from sklearn.metrics import confusion_matrix
cm_rfe = confusion_matrix(Y_test, Y_predict)
score_rfe = rfc2.score(X_test_rfe, Y_test)


# Get column names
columns = list(x.columns)

# Get the ranking of the features. Ranking 1 for selected features
ranking = rfe.ranking_

# Get the feature importance scores
feature_importance = rfc1.feature_importances_

# Create the dataframe of the Features selected, Ranking and their importance
rfe_selected = pd.DataFrame()
rfe_selected = pd.concat([pd.DataFrame(columns), 
                          pd.DataFrame(ranking),
                          pd.DataFrame(feature_importance)], axis=1)

rfe_selected.columns = ["Feature Name", "Ranking", "Feature Importance"]





