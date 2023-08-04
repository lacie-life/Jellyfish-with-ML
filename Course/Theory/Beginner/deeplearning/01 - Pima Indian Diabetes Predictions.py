# ---------------------------------------------------------------
# Implement Neural Net using Keras Sequential model
# Predict the onset of Diabetes for the Pima Indians based on 
# the available diagnostic data
# ---------------------------------------------------------------

# Import and seed various random functions for same result 
from numpy.random import seed
from tensorflow import set_random_seed
seed(123)
set_random_seed(124)

# Import Pandas, Sequential and Dense from Keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Read the csv file
diabetes = pd.read_csv('diabetes.csv')
diabetes.isnull().sum(axis=0)

# Create X and Y variables
X = diabetes.iloc[:, 0:-1]
Y = diabetes.iloc[:,   -1]


# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.2, random_state = 1234, stratify=Y)


# Define the keras sequential model with three hidden layers
model = Sequential()
model.add(Dense(24, 
                input_shape=(8,), 
                activation='relu', 
                kernel_initializer='RandomNormal'))

model.add(Dense(12, 
                activation='relu',
                kernel_initializer='RandomNormal'))

model.add(Dense(1, activation='sigmoid'))


# Compile the keras model for classification accuracy
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Fit the model on the training dataset
model.fit(X_train, Y_train, epochs=160, batch_size=10)


# Evaluate and print the model accuracy on test dataset
accuracy_test = model.evaluate(X_test, Y_test)
print("")
print("Test Accuracy     : " + str(accuracy_test[1]))


# Predict the classes using test data and the compiled model
Y_predict = model.predict_classes(X_test)
Y_pred_prob = model.predict(X_test)

# Create the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)




