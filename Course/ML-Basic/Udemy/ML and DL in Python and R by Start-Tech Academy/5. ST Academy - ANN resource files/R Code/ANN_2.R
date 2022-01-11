
#Install the keras R package
install.packages("keras")

#Install the core Keras library + TensorFlow
library(keras)
install_keras()

#install_keras(tensorflow = "gpu")

############

fashion_mnist <- dataset_fashion_mnist()

#Test Train Split
#train_images <- fashion_mnist$train$x
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Explore data structure
dim(train_images)
str(train_images)

#Plotting the image
fobject <- train_images[9,,]
plot(as.raster(fobject, max = 255))

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

class_names[train_labels[9]+1]


#Normalizing [(X-mean)/Std.Dev]

train_images <- train_images / 255
test_images <- test_images / 255


#Creating a validation split - used for hyperparameter tuning
val_indices <- 1:5000
val_images <- train_images[val_indices,,]
part_train_images <- train_images[-val_indices,,]
val_labels <- train_labels[val_indices]
part_train_labels <- train_labels[-val_indices]


# Flattening
# X X X
# Y Y Y  -> X X X Y Y Y Z Z Z
# Z Z Z

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'sgd', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)
# Sparse_categorical_crossentropy => more than 2 classes and observation can belong to only one class
# Binary_crossentropy => 2 classes and object belongs to one of the two classes
# Categorical_crossentropy => more than 2 classes and observation can belong to multiple classes


model %>% fit(part_train_images, part_train_labels, epochs = 30, batch_size=100, validation_data=list(val_images,val_labels))



# Test Performance

score <- model %>% evaluate(test_images, test_labels)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")

# Predicting on Test set

predictions <- model %>% predict(test_images)
predictions[1, ]
which.max(predictions[1, ])
class_names[which.max(predictions[1, ])]
plot(as.raster(test_images[1, , ], max = 1))

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]







# NeuralNet Package

# install package
install.packages("neuralnet")
require(neuralnet)

hours = c(20,10,30,20,50,30)
mocktest = c(90,20,20,10,50,80)
Passed = c(1,0,0,0,1,1)

df=data.frame(hours,mocktest,Passed)

nn=neuralnet(Passed~hours+mocktest,data=df, hidden=c(3,2),act.fct = "logistic", linear.output = FALSE)

plot(nn)

thours = c(20,20,30)
tmocktest = c(80,30,20)
test=data.frame(thours,tmocktest)
Predict=compute(nn,test)
Predict$net.result
prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)
pred








###### Regression Neural Network with Functional API

# Loading the inbuilt Dataset
boston_housing <- dataset_boston_housing()

# To know more about dataset: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

# Test data is *not* used when calculating the mean and std.

# Normalize training data
train_data <- scale(train_data) 

# Use means and standard deviations from training set to normalize test set
col_means_train <- attr(train_data, "scaled:center") 
col_stddevs_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)


# Functional API has two parts: inputs and outputs

# input layer
inputs <- layer_input(shape = dim(train_data)[2])

# outputs compose input + dense layers
predictions <- inputs %>%
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 1)

# create and compile model
model <- keras_model(inputs = inputs, outputs = predictions)
model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse',
  metrics = list("mean_absolute_error")
)

model %>% fit(train_data, train_labels, epochs = 30, batch_size=100)

# Test Performance

score <- model %>% evaluate(test_data, test_labels)
cat('Test loss:', score$loss, "\n")
cat('Test absolute error:', score$mean_absolute_error, "\n")


#------------------#

# input layer

inputs_func <- layer_input(shape = dim(train_data)[2])

# outputs compose input + dense layers

predictions_func <- inputs_func %>%
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 64, activation = 'relu') 

#Re-using the input features after the second hidden layer

main_output <- layer_concatenate(c(predictions_func, inputs_func)) %>%
  layer_dense(units = 1)


# create and compile model

model_func <- keras_model(inputs = inputs_func, outputs = main_output)
model_func %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse',
  metrics = list("mean_absolute_error")
)

summary(model_func)

model_func %>% fit(train_data, train_labels, epochs = 30, batch_size=100)

# Test Performance

score_func <- model_func %>% evaluate(test_data, test_labels)
cat('Functional Model Test loss:', score_func$loss, "\n")
cat('Normal model Test loss:', score$loss, "\n")
cat('Functional Model Test Mean Abs Error:', score_func$mean_absolute_error, "\n")
cat('Normal Model Test Mean Abs Error:', score$mean_absolute_error, "\n")






##### Saving and Restoring Models #####

#model_func %>% fit(train_data, train_labels, epochs = 30, batch_size=100)

model_func %>% save_model_hdf5("my_model.h5")

new_model <- load_model_hdf5("my_model.h5")

model_func %>% summary()
# or you can write summary(model)
new_model %>% summary()

### Using callbacks to create epoch store points

checkpoint_dir <- "checkpoints"
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "Epoch-{epoch:02d}.hdf5")

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(filepath = filepath)

rm(model_func)
k_clear_session()

model_callback <- keras_model(inputs = inputs_func, outputs = main_output)
model_callback %>% compile(optimizer = 'rmsprop',loss = 'mse',
                           metrics = list("mean_absolute_error"))

model_callback %>% fit(train_data, train_labels, epochs = 30,
                       callbacks = list(cp_callback))

list.files(checkpoint_dir)

tenth_model <- load_model_hdf5(file.path(checkpoint_dir, "Epoch-10.hdf5"))

summary(tenth_model)


##### Only saving the best model

callbacks_best <- callback_model_checkpoint(filepath = "best_model.h5", monitor = "val_loss", 
                                            save_best_only = TRUE)

rm(model_callback)
k_clear_session()

model_cb_best <- keras_model(inputs = inputs_func, outputs = main_output)
model_cb_best %>% compile(optimizer = 'rmsprop',loss = 'mse',
                          metrics = list("mean_absolute_error"))

model_cb_best %>% fit(train_data, train_labels, epochs = 30, 
                      validation_data=list(test_data,test_labels),
                      callbacks = list(callbacks_best))

best_model <- load_model_hdf5("best_model.h5")


### Stopping the processing when we find the best model

callbacks_list <- list(
  callback_early_stopping(monitor = "val_loss",patience = 3),
  callback_model_checkpoint(filepath = "best_model_early_stopping.h5", monitor = "val_loss", save_best_only = TRUE)
)

rm(model_cb_best)
k_clear_session()

model_cb_early <- keras_model(inputs = inputs_func, outputs = main_output)
model_cb_early %>% compile(optimizer = 'rmsprop',loss = 'mse',
                           metrics = list("mean_absolute_error"))

model_cb_early %>% fit(train_data, train_labels, epochs = 100, 
                       validation_data=list(test_data,test_labels),
                       callbacks = callbacks_list)

best_model_early_stopping <- load_model_hdf5("best_model_early_stopping.h5")

k_clear_session()
