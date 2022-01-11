# Compare the effect of Lasso, Ridger and Linear Regression
# for presence of Multicollinearity as well as their effect on
# coefficients of different features

# Import Pandas for data processing
import pandas as pd

# Read the CSV file
dataset = pd.read_csv('mcl.csv')
df = dataset.copy()

# Split into X (Independent) and Y (predicted)
X = df.iloc[:, :-1]
Y = df.iloc[:,  -1]

# Check and confirm the presence of Multicollinearity
correlation = df.corr()

# Import all the regressions to compare
from sklearn.linear_model import Lasso, Ridge, LinearRegression

# Perform Linear Regression 
lr = LinearRegression()
lr.fit(X, Y)

lr_coeff = lr.coef_
lr_intercept = lr.intercept_

# Perform Lasso regression
lasso = Lasso(alpha=10)
lasso.fit(X, Y)

lasso_coeff = lasso.coef_
lasso_intercept = lasso.intercept_

# Perform Ridge Regression
ridge = Ridge(alpha=100)
ridge.fit(X, Y)

ridge_coeff = ridge.coef_
ridge_intercept = ridge.intercept_

# At the end compare the values of three lists of coefficients





