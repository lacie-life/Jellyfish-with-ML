

### Installing and activating the libraries

install.packages("keras")

##devtools::install_github("rstudio/keras")

##Import the Fashion MNIST dataset

library(keras)
install_keras(tensorflow = cpu)

install.packages("tensorflow")
library(keras)

#library("jpeg")
#jj <- readJPEG("CNN.jpg",native=TRUE)
#plot(0:1,0:1,type="n",ann=FALSE,axes=FALSE)
#rasterImage(jj,0,0,1,1)

#importing data
fashion_mnist <- dataset_fashion_mnist()

#Test Train Split
#train_images <- fashion_mnist$train$x
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

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

class_names[train_labels[5]+1]


##Preprocess the data


train_images <- train_images / 255
test_images <- test_images / 255

val_indices <- 1:5000
val_images <- train_images[val_indices,,]
part_train_images <- train_images[-val_indices,,]
val_labels <- train_labels[val_indices]
part_train_labels<- train_labels[5001:60000]

str(part_train_images)
part_train_images <- array_reshape(part_train_images, c(55000, 28, 28, 1))
val_images <- array_reshape(val_images, c(5000, 28, 28, 1))
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))


##Define model architecture

model <- keras_model_sequential() %>%
          layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",input_shape = c(28, 28,1))
#%>%
          #layer_max_pooling_2d(pool_size = c(2, 2)) 
  #%>%layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  #layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  #layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

model <- model %>%
  layer_flatten() %>%
  layer_dense(units = 300, activation = "relu") %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = 10, activation = "softmax")

model


# Configuring the Model

model %>% compile(
  optimizer = 'sgd', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)
# Sparse_categorical_crossentropy => more than 2 classes and observation can belong to only one class
# Binary_crossentropy => 2 classes and object belongs to one of the two classes
# Categorical_crossentropy => more than 2 classes and observation can belong to multiple classes


model %>% fit(part_train_images, part_train_labels, epochs = 10, batch_size=64, validation_data=list(val_images,val_labels))



# Test Performance

CNN_score <- model %>% evaluate(test_images, test_labels)

cat('Test loss:', CNN_score$loss, "\n")
cat('Test accuracy:', CNN_score$acc, "\n")

# Predicting on Test set

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]
class_names[class_pred[1:20]+1]
class_names[test_labels[1:20]+1]
plot(as.raster(test_images[1, , , ]), max = 255)


