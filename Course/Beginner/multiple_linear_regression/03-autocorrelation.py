# --------------------------------------------------------------
# Create the Autocorrelation Plot and create time-lagged dataset
# --------------------------------------------------------------

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read the corr.csv file
f = pd.read_csv('03 - corr.csv')

# Convert the datatype of the variable to float
f['t0'] = pd.to_numeric(f['t0'], downcast='float')

# Plot the autocorrelation plot
plt.acorr(f['t0'], maxlags=10)


# Create the time-lagged dataset and concatenate the results
t_1 = f['t0'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = f['t0'].shift(+2).to_frame()
t_2.columns = ['t-2']

result = pd.concat([f['t0'], t_1, t_2], axis=1)
