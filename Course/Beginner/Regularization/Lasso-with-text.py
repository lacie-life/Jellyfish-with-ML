# Implement and see the effect of Lasso Regression 
# using plots of different Lasso regression lines

# Import Pandas for data processing
import pandas as pd

# Read the CSV file
dataset = pd.read_csv('ridge.csv')
df = dataset.copy()

# Split into X (Independent) and Y (predicted)
X = df.iloc[:, :-1]
Y = df.iloc[:,  -1]

# Import Lasso and matplotlib
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Create test data for plotting
X_plt = [0,1,2,3,4]


# Create a list of different alpha/penalty parameter values
lasso_l = [0,0.5,1,2,4]

# Plot different Lasso Regression lines in one figure
for i, l in enumerate(lasso_l):
    lasso = Lasso(alpha=l)
    lasso.fit(X, Y)

    lasso_coeff = lasso.coef_
    lasso_intercept = lasso.intercept_
    Y_plt = lasso.predict(pd.DataFrame(X_plt))
    
    plt.figure(1)       
    plt.plot(X_plt, Y_plt)
    plt.ylim(ymin=0, ymax=9)
    plt.xlim(xmin=0, xmax=6)
    plt.text(X_plt[-1], Y_plt[-1],  
             ' y = ' + 
             str('%.2f' %lasso_coeff) +
             ' * x' + 
             ' + ' + 
             str('%.2f' %lasso_intercept) +
             '      for \u03BB or \u03B1 = ' + str(l), fontsize=12)




