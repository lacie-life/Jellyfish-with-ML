import pandas as pd

# read csv
data_set = pd.read_csv('loan-small.csv')

# Access the dataframe data using iloc
subset = data_set.iloc[0:3, 1:3]

# Access the data using column names
subnetN = data_set[['Gender', 'ApplicantIncome']][0:3]

# Read the TSV file using pandas
datasetT = pd.read_csv('loan-small-tsv.txt', sep='\t')

data_set.head()
data_set.shape
data_set.columns

# Find out the columns with missing values
data_set.isnull().sum(axis=0)

# Data preprocessing by replacing the missing values
# drop the rows with the missing values
cleandata = data_set.dropna()

# copy the dataset
dt = data_set.copy()

# Replace categorical values with mode
cols = ['Gender', 'Area', 'Loan_Status']
dt[cols] = dt[cols].fillna(dt.mode().iloc[0])
dt.isnull().sum(axis=0)

# Eeplace numberical values with mean
cols2 = ['ApplicatIncome', 'CoapplicantIncome', 'LocanAmount']
dt[cols2] = dt[cols2].fillna(dt.mean())
dt.isnull().sum(axis=0)


