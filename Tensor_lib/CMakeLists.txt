project(Tensor)

set(HEADER_FILES
        Tensor.h
        TensorLoader.h
        )

set(SOURCE_FILES
        Tensor.cpp
        TensorLoader.cpp
        )

add_library(Tensor_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})