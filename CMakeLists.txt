cmake_minimum_required(VERSION 3.15)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES
        main.cpp)

add_executable(NeuralNet ${SOURCE_FILES})

include_directories(StringTools_lib)
add_subdirectory(StringTools_lib)

include_directories(Tensor_lib)
add_subdirectory(Tensor_lib)

include_directories(DataManager_lib)
add_subdirectory(DataManager_lib)

include_directories(Network_lib)
add_subdirectory(Network_lib)

target_link_libraries(NeuralNet StringTools_lib Tensor_lib DataManager_lib Network_lib)

add_subdirectory(Google_tests)