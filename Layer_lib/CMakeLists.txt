project(Layer)

set(HEADER_FILES
        Layer.h
        DenseLayer.h
        )

set(SOURCE_FILES
        DenseLayer.cpp
        )

add_library(Layer_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})