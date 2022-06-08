cmake_minimum_required(VERSION 3.15)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)

if (WIN32)
    set(CMAKE_CXX_FLIGS "${CMAKE_CXX_FLAGS} -std=c++17 -lmingw32")
    set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
endif (WIN32)

set(SOURCE_FILES
        main.cpp Simulation.cpp Simulation.h)

add_executable(NeuralNet ${SOURCE_FILES})

include_directories(Tensor_lib)
add_subdirectory(Tensor_lib)

include_directories(DataManager_lib)
add_subdirectory(DataManager_lib)

include_directories( StringTools_lib )
add_subdirectory( StringTools_lib )

include_directories(Network_lib)
add_subdirectory(Network_lib)

if (WIN32)
    target_include_directories(NeuralNet PUBLIC "SDL2/x86_64-w64-mingw32/include/SDL2")
    target_link_directories(NeuralNet PUBLIC "SDL2/x86_64-w64-mingw32/lib")

    target_link_libraries(NeuralNet mingw32 SDL2main SDL2 StringTools_lib Tensor_lib DataManager_lib Network_lib )
endif (WIN32)

add_subdirectory(Google_tests)