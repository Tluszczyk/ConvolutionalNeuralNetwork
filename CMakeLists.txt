cmake_minimum_required(VERSION 3.15)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES
        main.cpp
        Sequential.cpp
        Sequential.h
        LossFunctionsProvider.cpp LossFunctionsProvider.h)
add_executable(NeuralNet ${SOURCE_FILES})

include_directories(Tensor_lib)
add_subdirectory(Tensor_lib)

include_directories(Layer_lib)
add_subdirectory(Layer_lib)

target_link_libraries(NeuralNet Tensor_lib Layer_lib)

add_subdirectory(Google_tests)