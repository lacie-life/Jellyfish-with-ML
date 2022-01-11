import pandas as pd

# read csv
data_set = pd.read_csv('loan-small.csv')

cleandata = data_set.dropna()

# Extract the three numerical columns
data_to_scale = cleandata.iloc[:, 2:5]

# import standard scaler

from sklearn.preprcessing import StandardScaler

scaler_ = StandardScaler()
ss_scaler = scaler_.fit_transform(data_to_scale)

# minmax scaler
from sklearn.preprcessing import minmax_scale
mm_scaler = minmax_scale(data_to_scale)



