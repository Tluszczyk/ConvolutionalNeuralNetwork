project(Layer)

target_include_directories(NeuralNet PUBLIC "SDL2/x86_64-w64-mingw32/include/SDL2")
target_link_directories(NeuralNet PUBLIC "SDL2/x86_64-w64-mingw32/lib")



set(HEADER_FILES
        ActivationFunctionsProvider.h
        LossFunctionsProvider.h
        Sequential.h
        ModelLoader.h
        Layers/Layer.h
        Layers/DenseLayer.h
        Layers/LayerLoader.h
        Layers/LayerType.h
        Layers/OutputLayer.h)

set(SOURCE_FILES
        ActivationFunctionsProvider.cpp
        Sequential.cpp
        LossFunctionsProvider.cpp
        ModelLoader.cpp
        Layers/DenseLayer.cpp
        Layers/LayerLoader.cpp
        Layers/OutputLayer.cpp)




add_library(Network_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries( Network_lib
        Tensor_lib StringTools_lib )