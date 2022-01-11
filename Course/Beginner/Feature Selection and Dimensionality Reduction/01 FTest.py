# --------------------------------------------------------
# Apply Feature Selection with F-Test on Linear Regression
# Compare the result with selected features
# --------------------------------------------------------

# Import libraries
import pandas as pd

# Read the file
f = pd.read_csv('Students2.csv')

# Split the columns into Dependent (Y) and independent (X) features
x = f.iloc[:,:-1]
y = f.iloc[:, -1]

# Perform Linear Regression using original dataset
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(x, y, test_size = 0.4, random_state = 1234)

lr.fit(X_train, Y_train)

y_predict = lr.predict(X_test)

# Calculate the RMSE error for the regression
from sklearn.metrics import mean_squared_error
import math

rmse = math.sqrt(mean_squared_error(Y_test, y_predict))


# import and perform the f_regression to get the F-Score and P-Values
from sklearn.feature_selection import f_regression as fr
result = fr(x,y)


# Split the result tuple into F_Score and P_Values
f_score = result[0]
p_values = result[1]


# Print the table of Features, F-Score and P-values
columns = list(x.columns)

print (" ")
print (" ")
print (" ")

print ("    Features     ", "F-Score    ", "P-Values")
print ("    -----------  ---------    ---------")

for i in range(0, len(columns)):
    f1 = "%4.2f" % f_score[i]
    p1 = "%2.6f" % p_values[i]
    print("    ", columns[i].ljust(12), f1.rjust(8),"    ", p1.rjust(8))


# Perform the Linear Regression with reduced features
X_train_n = X_train[['Hours', 'sHours']]
X_test_n = X_test[['Hours', 'sHours']]

lr1 = LinearRegression()
lr1.fit(X_train_n, Y_train)

y_predict_n = lr1.predict(X_test_n)

# Calculate the RMSE with reduced features
rmse_n = math.sqrt(mean_squared_error(Y_test, y_predict_n))








