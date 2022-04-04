cmake_minimum_required(VERSION 3.12)
project(reading)
SET(CMAKE_CXX_STANDARD 20)
if(WIN32)
add_compile_options("-march=native")
add_compile_options("-O2")
else()
SET(CMAKE_CXX_FLAGS "-Ofast -mtune=native -lstdc++fs -march=native -pthread -flto -ffat-lto-objects -fuse-linker-plugin")
endif()
SET(CMAKE_BUILD_TYPE "Release")

if (UNIX)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Training")
    file(GLOB THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.h ${CMAKE_CURRENT_SOURCE_DIR}/Training/Util/*.h  )
    file(GLOB TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Training/Util/*.cpp)
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


add_executable(MainEngine ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources} Checkers/CheckerEngineX/Network.cpp Checkers/CheckerEngineX/Network.h Checkers/CheckerEngineX/Thread.cpp Checkers/CheckerEngineX/Thread.h)

add_executable(perft_check ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Checks/perft_check.cpp ${CHeaders} ${CSources})

if (WIN32)
    add_executable(tb_rescorer ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Checks/tb_rescorer.cpp ${CHeaders} ${CSources} ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb.h ${CMAKE_CURRENT_SOURCE_DIR}/Training/BloomFilter.h ${CMAKE_CURRENT_SOURCE_DIR}/Training/Sample.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Training/Sample.h )
endif ()

add_library(pyhelper SHARED ${PyHeaders} ${PySources} ${CSources} ${CHeaders} ${TSources} ${THeaders} Training/Util/LRUCache.h)


if (UNIX)
    add_executable(Train ${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp ${THeaders} ${TSources} ${CHeaders} ${CSources} Training/Generator.cpp Training/Generator.h Training/BloomFilter.h Training/Sample.cpp Training/Sample.h Training/PosStreamer.cpp Training/PosStreamer.h Training/BatchProvider.cpp Training/BatchProvider.h)
    set_target_properties(Train PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
endif ()
set_target_properties(MainEngine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
set_target_properties(perft_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
if (UNIX)
    target_compile_definitions(Train PUBLIC "TRAIN")
endif ()
if (WIN32)
      target_link_libraries(tb_rescorer  ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib)
endif ()

