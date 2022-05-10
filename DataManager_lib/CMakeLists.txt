project(DataManager)

set(HEADER_FILES
        DataPrep.h
        DataProvider.h
        )

set(SOURCE_FILES
        DataPrep.cpp
        DataProvider.cpp
        )

add_library(DataManager_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})