project(Layer)

set(HEADER_FILES
        ActivationFunctionsProvider.h
        LossFunctionsProvider.h
        Sequential.h
        ModelLoader.h
        Layers/Layer.h
        Layers/DenseLayer.h
        Layers/LayerLoader.h
        Layers/LayerType.h
        Layers/OutputLayer.h
        )

set(SOURCE_FILES
        ActivationFunctionsProvider.cpp
        Sequential.cpp
        LossFunctionsProvider.cpp
        ModelLoader.cpp
        Layers/DenseLayer.cpp
        Layers/LayerLoader.cpp
        Layers/OutputLayer.cpp
        )

add_library(Network_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})