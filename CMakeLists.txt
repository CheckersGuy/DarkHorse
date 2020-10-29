cmake_minimum_required(VERSION 3.12)
project(reading)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Ofast -DNDEBUG -march=native -pthread  -flto -ffat-lto-objects -fuse-linker-plugin ")


if (UNIX)
    find_package(Protobuf REQUIRED)
    include_directories(${PROTOBUF_INCLUDE_DIR})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Training")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Training/proto")


    file(GLOB PROTOBUF_MODELS_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/Training/proto/*.pb.cc" "${CMAKE_CURRENT_SOURCE_DIR}/Training/proto/*.hpp")
    file(GLOB THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.h)
    file(GLOB TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.cpp)
    file(GLOB PHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/proto/*.pb.h)
    file(GLOB PSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/proto/*.pb.cc)


    list(REMOVE_ITEM TSources "${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp")
endif ()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX")
file(GLOB CHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/*.h)
file(GLOB CSources ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/*.cpp)

list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp")



add_executable(MainEngine ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources})
add_executable(perft_check ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Checks/perft_check.cpp ${CHeaders} ${CSources})
add_executable(Generator ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources})
if (UNIX)
    add_executable(Train ${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp ${THeaders} ${TSources} ${CHeaders} ${CSources} ${PSources} ${PHeaders} Training/HyperLog.h)
    set_target_properties(Train PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
endif ()

set_target_properties(Generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Training/Engines")
set_target_properties(MainEngine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
set_target_properties(perft_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")

if (UNIX)
    target_link_libraries(Train ${PROTOBUF_LIBRARY})
    target_compile_definitions(Train PUBLIC "TRAIN")
endif ()


target_compile_definitions(Generator PUBLIC "GENERATOR")