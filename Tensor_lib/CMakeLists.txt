project(Tensor)

set(HEADER_FILES
        Tensor.h
        )

set(SOURCE_FILES
        Tensor.cpp
        )

add_library(Tensor_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})