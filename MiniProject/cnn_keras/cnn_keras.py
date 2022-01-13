import numpy as np
import math 
from matplotlib import pyplot as plt
from mnist import MNIST
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

def showInputData(input_Image,r,c):
    plt.figure()
    for i in range(1,r*c+1):
        plt.subplot(r,c,i)
        plt.axis('off')
        plt.grid(b=None)
        plt.imshow(input_Image[i-1])
    plt.show()

num_trainData = 60000
num_testData = 950
num_test = 9
num_showData_window = math.floor(math.sqrt(num_test))
num_filters = 8
filter_size = 3
pool_size = 2

# --------------------------------- DATA PRE-PROCESSING ----------------------------
mndata = MNIST('/Users/lufan/OneDrive/Machine Learning/CNN/handWrittenDigit_MNIST/python-mnist/data')
mndata.gz = False
load_train_images, train_labels = mndata.load_training()[:num_trainData] 
load_test_images, test_labels = mndata.load_testing()[:num_testData]

# Load input images for training set and validation set
train_images = np.array(load_train_images[:num_trainData])
train_images = np.reshape(train_images,(num_trainData,28,28)).astype(int)
test_images = np.array(load_test_images[:num_testData])
test_images = np.reshape(test_images,(num_testData,28,28)).astype(int)

# Normalize and expand dimension for input images
train_images = (train_images / 255) - 0.5
train_images = np.expand_dims(train_images, axis=3)
test_images = (test_images/ 255) - 0.5
test_images = np.expand_dims(test_images, axis=3)

# Load labels for training set and validation set
train_labels = train_labels[:num_trainData]
train_labels = np.array(train_labels)
test_labels = test_labels[:num_testData]
test_labels = np.array(test_labels)

# --------------------------------------- MODEL TRAINING ------------------------------
# Model init
model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28,28,1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax'),
])
# Compile model
model.compile(
    'adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
# print(train_images.shape, train_labels.shape)
# Train model
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=10,
    validation_data = (test_images[:num_testData-num_test], to_categorical(test_labels[:num_testData-num_test])),
)

# # Save the trained model to disk
# model.save_weights('cnn1.h5')

# ------------------------------------ PREDICTION -------------------------------------
# Load the pre-trained model
# model.load_weights('cnn.h5')

# Prediction using the trained model
# test = np.array(test_images[num_testData-num_test])
# for i in range (1,num_test):
#     test = np.append(test,test_images[num_testData+i-num_test])
test = test_images[num_testData-num_test : num_testData]

predict = model.predict(test)
print("Labels: ", test_labels[num_testData-num_test:num_testData])
print("Predicted:", end=" ")
count = 0
for i in range (num_test):
    if test_labels[num_testData-num_test+i] != np.argmax(predict[i]): 
        count += 1
        print(np.argmax(predict[i]),'(',test_labels[num_testData-num_test+i],')', end=" ")
    else:
        print(np.argmax(predict[i]),end=" ")
print("\n","Errors: ", count)
# showInputData(test,num_showData_window,num_showData_window)
plt.imshow(test[1])
plt.show()