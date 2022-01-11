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

# Label Encodeing using Python
dt.dtypes

# Changes the datatypes
dt[cols] = dt[cols].astype('category')

for columns in cols:
    dt[columns] = dt[columns].cat.codes

# Create Dummy variables or Hot encoding
df2 = data_set.drop(['Loan_ID'], axis=1)
df2 = pd.get_dummies(df2)

# Split the dât vertically into X and Y
X = df2.iloc[:, :-1]  # Select all yhe columns execpt loan_status
Y = df2.iloc[:, -1]   # Select only the loan_status


# Split the dataset by rows
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

