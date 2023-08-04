# -----------------------------------------------------------------
# Compare multiple Classifiers and tune the hyperparameters 
# using GridSearchCV
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

# Import and create Support Vector Classifier
from sklearn.svm import SVC
svc = SVC(random_state=1234)

# Import and create Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(random_state=1234)


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define parameters for Random Forest
rfc_param = {'n_estimators':[10,15,20], 
            'min_samples_split':[8,16],
            'min_samples_leaf':[1,2,3,4,5]
            }

# The parameters results in 3 x 2 x 5 = 30 different combinations
# CV=10 for 30 different combinations mean 300 jobs/model runs

rfc_grid = GridSearchCV(estimator=rfc, 
                        param_grid=rfc_param,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1,
                        return_train_score=True)

# Fit the data to do Grid Search
rfc_grid_fit = rfc_grid.fit(X, Y)

# Get the results of the GridSearchCV
cv_results_rfc = pd.DataFrame.from_dict(rfc_grid_fit.cv_results_)



# Define parameters for Logistic Regression
lrc_param = {'C':[0.01, 0.1, 0.5, 1, 2, 5, 10], 
            'penalty':['l2'],
            'solver':['liblinear','lbfgs', 'saga']
            }

# The parameters results in 7 x 1 x 3 = 21 different combinations
# CV=10 for 21 different combinations mean 210 jobs/model runs

lrc_grid = GridSearchCV(estimator=lrc, 
                        param_grid=lrc_param,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1,
                        return_train_score=True,
                        verbose=True)

# Fit the data to do Grid Search using Logistic Regression
lrc_grid_fit = lrc_grid.fit(X, Y)

# Get the Grid Search results for Logistic Regression
cv_results_lrc = pd.DataFrame.from_dict(lrc_grid_fit.cv_results_)


# define parameters for Support Vector Classifier
svc_param = {'C':[0.01, 0.1, 0.5, 1, 2, 5, 10], 
            'kernel':['rbf', 'linear'],
            'gamma':[0.1, 0.25, 0.5, 1, 5]
            }

# The parameters results in 7 x 2 x 5 = 70 different combinations
# CV=10 for 70 different combinations mean 700 jobs/model runs

svc_grid = GridSearchCV(estimator=svc, 
                        param_grid=svc_param,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1,
                        return_train_score=True)

# Fit the data to do Grid Search for Support Vector
svc_grid_fit = svc_grid.fit(X, Y)

# Get the Grid Search results for Support Vector
cv_results_svc = pd.DataFrame.from_dict(svc_grid_fit.cv_results_)

# Get the top ranked test score for all the three classifiers
rfc_top_rank = cv_results_rfc[cv_results_rfc['rank_test_score'] == 1]
lrc_top_rank = cv_results_lrc[cv_results_lrc['rank_test_score'] == 1]
svc_top_rank = cv_results_svc[cv_results_svc['rank_test_score'] == 1]


# Print the train and test score for three classifiers

print('\n\n')

print ('                    ',
       '  Random Forest    ',
       '  Logistic Regression  ',
       '  Support Vector   ')

print ('                    ',
       '  ---------------- ',
       '  -------------------- ',
       '  ---------------- ')


print ('  Mean Test Score   : ', 
       str('%.4f' %rfc_top_rank['mean_test_score']),
       '            ',
       str('%.4f' %lrc_top_rank['mean_test_score']),
       '                ',
       str('%.4f' %svc_top_rank['mean_test_score'])
       )

print ('  Mean Train Score  : ', 
       str('%.4f' %rfc_top_rank['mean_train_score']),
       '            ',
       str('%.4f' %lrc_top_rank['mean_train_score']),
       '                ',
       str('%.4f' %svc_top_rank['mean_train_score'])
       )

# Print the best parameters of the Random Forest Classifier
print('\n The best Parameters are : ')
print(rfc_grid_fit.best_params_)



