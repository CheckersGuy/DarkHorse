cmake_minimum_required(VERSION 3.12)
project(reading)
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS "-Ofast -pthread -fprofile-use -march=native -flto -ffat-lto-objects -fuse-linker-plugin")
SET(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE Debug)
if (UNIX)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Training")
    file(GLOB THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.h)
    file(GLOB TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.cpp)
    file(GLOB PySources ${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.cpp)
    file(GLOB PyHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.h)
    list(REMOVE_ITEM TSources "${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp")
    list(REMOVE_ITEM TSources "${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.cpp")
    list(REMOVE_ITEM THeaders "${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.h")
endif ()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX")
file(GLOB CHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/*.h)
file(GLOB CSources ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/*.cpp)

list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp")


add_executable(MainEngine ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources} Checkers/CheckerEngineX/Network.cpp Checkers/CheckerEngineX/Network.h)

add_executable(perft_check ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Checks/perft_check.cpp ${CHeaders} ${CSources})
add_library(pyhelper SHARED ${PyHeaders} ${PySources} ${CSources} ${CHeaders})


if (UNIX)
    add_executable(Train ${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp ${THeaders} ${TSources} ${CHeaders} ${CSources}  Training/HyperLog.h Training/Generator.cpp Training/Generator.h Training/GeneratorZ.cpp Training/GeneratorZ.h)
    set_target_properties(Train PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
endif ()
set_target_properties(MainEngine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
set_target_properties(perft_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
if (UNIX)
    target_compile_definitions(Train PUBLIC "TRAIN")
endif ()

