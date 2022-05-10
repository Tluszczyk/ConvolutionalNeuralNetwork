cmake_minimum_required(VERSION 3.15)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)

#set(CMAKE_CXX_FLIGS "${CMAKE_CXX_FLAGS} -std=c++17")
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

set(SOURCE_FILES
        main.cpp Simulation.cpp Simulation.h)

add_executable(NeuralNet ${SOURCE_FILES})

include_directories(StringTools_lib)
add_subdirectory(StringTools_lib)

include_directories(Tensor_lib)
add_subdirectory(Tensor_lib)

include_directories(DataManager_lib)
add_subdirectory(DataManager_lib)

include_directories(Network_lib)
add_subdirectory(Network_lib)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIR})


target_link_libraries(NeuralNet StringTools_lib Tensor_lib DataManager_lib Network_lib)

#target_include_directories(NeuralNet PUBLIC "SDL2/x86_64-w64-mingw32/include/SDL2")
#target_link_directories(NeuralNet PUBLIC "SDL2/x86_64-w64-mingw32/lib")


add_subdirectory(Google_tests)