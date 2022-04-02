project(Layer)

set(HEADER_FILES
        Layer.h
        DenseLayer.h
        LayerLoader.h
        LayerType.h
        ActivationFunctionsProvider.h
        )

set(SOURCE_FILES
        DenseLayer.cpp
        LayerLoader.cpp
        ActivationFunctionsProvider.cpp
        )

add_library(Layer_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})