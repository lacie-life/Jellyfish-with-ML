# ----------------------------------------------------------------------
# Implement Principal Component Analysis (PCA) for the Breast Cancer 
# prediction and compare results
# ----------------------------------------------------------------------

# Import libraries, load the dataset and create X and Y
from sklearn.datasets import load_breast_cancer
import pandas as pd

lbc = load_breast_cancer()

X = pd.DataFrame(lbc['data'], columns=lbc['feature_names'])
Y = pd.DataFrame(lbc['target'], columns=['type'])

# --------------------------------------
# Perform the prediction Without PCA
# --------------------------------------

# Split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)

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


# ----------------------------------
# Perform PCA and compare results 
# ----------------------------------

# Normalize the data with mean as zero
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

# Check the mean of the centered data
X_scaled[:,0].mean()

# Import PCA and fit the data to create PCAs
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)


# Split the dataset into train and test
X_train, X_test, Y_train, Y_test = \
train_test_split(X_pca, Y, test_size = 0.3, random_state = 1234, stratify=Y)


# Default Random Forest Object
rfc2 = RandomForestClassifier(random_state=1234)
rfc2.fit(X_train, Y_train)
Y_predict2 = rfc2.predict(X_test)


# Score and Evaluate the model using transformed data
cm2 = confusion_matrix(Y_test, Y_predict2)
score2 = rfc2.score(X_test, Y_test)













