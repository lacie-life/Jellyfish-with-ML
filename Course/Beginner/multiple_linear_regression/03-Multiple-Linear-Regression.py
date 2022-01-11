# ----------------------------------------------------------------
# Predict the marks obtained based on the Number of hours studied
# and the number of hours slept 
# ----------------------------------------------------------------

# Import Pandas for data processing
import pandas as pd


# Read the CSV file
dataset = pd.read_csv('02Students.csv')
df = dataset.copy()

# Split into X (Independent) and Y (predicted)
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]


# Split for rows
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =     \
train_test_split (X, Y, test_size = 0.3, random_state=1234)

# train the Simple Linear Regression
from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()

# Provide the training Data
std_reg.fit(x_train, y_train)

# predict the results
y_predict = std_reg.predict(x_test)

# Get the R-Squared 
mlr_score = std_reg.score(x_test, y_test)

# Coefficient and Intercept
mlr_coefficient = std_reg.coef_
mlr_intercept = std_reg.intercept_

# Equation of the line
#   y = 1.31  + 4.67*Hours + 5.1*SHours

# Calculate the errors using RMSE 
from sklearn.metrics import mean_squared_error
import math

mlr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))
















