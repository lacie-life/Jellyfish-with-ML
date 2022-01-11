# -----------------------------------------------------------------
# Perform RandomizedSearchCV for hyperparameter tuning
# -----------------------------------------------------------------

# Import libraries
import pandas as pd

# Read dataset
data = pd.read_csv('hpt_small.csv')

# Create Dummy variables
data_prep = pd.get_dummies(data, drop_first=True)

# Create X and Y Variables
X = data_prep.iloc[:, :-1]
Y = data_prep.iloc[:, -1]

# Import and create Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)

# Import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

# define parameters for Random Forest
rfc_param = {'n_estimators':[10,15,20], 
            'min_samples_split':[8,16],
            'min_samples_leaf':[1,2,3,4,5]
            }

# The parameters results in 3 x 2 x 5 = 30 different combinations

# Create the RandomizedSearchCV object
rfc_rs = RandomizedSearchCV(estimator=rfc, 
                        param_distributions=rfc_param,
                        scoring='accuracy',
                        cv=10,
                        n_iter=10,
                        return_train_score=True,
                        random_state=1234)

# n_iter selects 10 combinations out of 30 possible
# Now 10 x 10 = 100 jobs will be executed

# Fit the data to RandomizedSearchCV object
rfc_rs_fit = rfc_rs.fit(X, Y)

# Get the results of RandomizedSearch
cv_results_rfc_rs = pd.DataFrame.from_dict(rfc_rs_fit.cv_results_)

# Print the best parameters of Randomized Search for Random Forest
print('\n The best Parameters are : ')
print(rfc_rs_fit.best_params_)













