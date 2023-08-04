cmake_minimum_required(VERSION 2.8)

project(mnist_sample)

add_executable(mnist_sample sampleMNIST.cpp)

target_link_libraries(mnist_sample nvinfer nvinfer_plugin nvcaffe_parser)