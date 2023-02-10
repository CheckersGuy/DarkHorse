cmake_minimum_required(VERSION 3.12)
project(reading)
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_BUILD_TYPE "Release")
  add_compile_options("-mtune=native")
  add_compile_options("-lstdc++fs")
  add_compile_options("-march=native")
  add_compile_options("-pthread")
  add_compile_options("-ffat-lto-objects")
  add_compile_options("-fuse-linker-plugin")
  add_compile_options("-Ofast")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX")
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/Training/generator.proto)
#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/DESCRIPTORS Training/generator.proto)


if (UNIX)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Training)
    file(GLOB THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.h ${CMAKE_CURRENT_SOURCE_DIR}/Training/Util/*.h  )
    file(GLOB TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/Training/Util/*.cpp)
    file(GLOB PySources ${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.cpp)
    file(GLOB PyHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.h)
    list(REMOVE_ITEM TSources  ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.cpp)
    list(REMOVE_ITEM THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.h)
    list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.cpp)
    list(REMOVE_ITEM THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.h)
endif ()


file(GLOB CHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/*.h)
file(GLOB CSources ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/*.cpp)
list(REMOVE_ITEM CHeaders "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.h")
list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.cpp")
list(REMOVE_ITEM CHeaders "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb.h")
list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp")



add_executable(MainEngine ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources})



add_executable(perft_check ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Checks/perft_check.cpp ${CHeaders} ${CSources})
add_library(Dark SHARED ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.h ${CHeaders} ${CSources})


add_library(pyhelper SHARED ${PROTO_SRCS} ${PROTO_HDRS} ${PyHeaders} ${PySources} ${CSources} ${CHeaders} ${THeaders} ${TSources})
set_target_properties(pyhelper PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Training")


if (UNIX)
    add_executable(Train ${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp ${THeaders} ${TSources} ${CHeaders} ${CSources} ${PROTO_SRCS} ${PROTO_HDRS})
    set_target_properties(Train PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
endif ()
set_target_properties(MainEngine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
set_target_properties(perft_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")

#target_compile_definitions(MainEngine PUBLIC "USE_DB")
target_compile_definitions(Dark PUBLIC "USE_DB")
#target_link_libraries(MainEngine  ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib)
#target_link_libraries(Dark  ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib)
target_link_libraries(Train ${Protobuf_LIBRARIES})
target_link_libraries(pyhelper ${Protobuf_LIBRARIES})

