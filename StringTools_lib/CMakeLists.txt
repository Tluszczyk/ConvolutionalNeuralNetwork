project(StringTools)

set(HEADER_FILES
        StringTools.h
        )

set(SOURCE_FILES
        StringTools.cpp
        )

add_library(StringTools_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})