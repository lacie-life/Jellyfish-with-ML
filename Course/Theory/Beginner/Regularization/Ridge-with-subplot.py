# Implement and see the effect of Ridge Regression 
# using subplots of different Ridge regression lines

# Import Pandas for data processing
import pandas as pd

# Read the CSV file
dataset = pd.read_csv('ridge.csv')
df = dataset.copy()

# Split into X (Independent) and Y (predicted)
X = df.iloc[:, :-1]
Y = df.iloc[:,  -1]

# Import Ridge and matplotlib
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Create test data for plotting
X_plt = [0,1,2,3,4]

# Create a list of different alpha/penalty parameter values
ridge_l = [0,1,10,100]

# Create subplots for Ridge Regression lines in one figure
for i, l in enumerate(ridge_l):
    ridge = Ridge(alpha=l)
    ridge.fit(X, Y)

    ridge_coeff = ridge.coef_
    ridge_intercept = ridge.intercept_
    Y_plt = ridge.predict(pd.DataFrame(X_plt))

    plt.figure(1)    
    plt.subplot(2,2,i+1)
    plt.plot(X_plt, Y_plt)
    plt.ylim(ymin=0, ymax=9)
    plt.xlim(xmin=0, xmax=6)
    plt.title(' y = ' + 
             str('%.2f' %ridge_coeff) +
             ' * x' + 
             ' + ' + 
             str('%.2f' %ridge_intercept) +
             '      for \u03BB or \u03B1 = ' + str(l), fontsize=12)
    plt.tight_layout()







