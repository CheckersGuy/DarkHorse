cmake_minimum_required(VERSION 3.9)
project(reading)
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_BUILD_TYPE "Release")
  add_compile_options("-march=native")
  add_link_options("-flto")
  add_compile_options("-fuse-linker-plugin")
  add_compile_options("-Ofast")
  add_compile_options("-mavx2")
  add_compile_options("-mavx")
  add_compile_options("-funroll-loops")
  add_compile_options("-mpopcnt")
  add_compile_options("-msse")
  add_compile_options("-msse3")
  add_compile_options("-mpopcnt")
  add_compile_options("-msse4.1")
  add_compile_options("-msse2")
  add_link_options("-lstdc++")
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
  if(UNIX)
    include(CheckIPOSupported)
    check_ipo_supported()
    find_package(Protobuf REQUIRED)
    include_directories(${Protobuf_INCLUDE_DIRS})
  endif()
  if(WIN32)
    set(CMAKE_CXX_STANDARD_LIBRARIES "-static-libgcc -static-libstdc++ -lwsock32 -lws2_32 ${CMAKE_CXX_STANDARD_LIBRARIES}")
  endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX")


if (UNIX)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Training)
    file(GLOB THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.h ${CMAKE_CURRENT_SOURCE_DIR}/Training/Util/*.h  )
    file(GLOB TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/Training/Util/*.cpp)
    file(GLOB PySources ${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.cpp)
    file(GLOB PySources ${CMAKE_CURRENT_SOURCE_DIR}/Training/BatchProvider.cpp)
    file(GLOB PySources ${CMAKE_CURRENT_SOURCE_DIR}/Training/PosStreamer.cpp)
    file(GLOB PySources ${CMAKE_CURRENT_SOURCE_DIR}/Training/Sample.cpp)
    file(GLOB PySources ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Position.cpp)




    file(GLOB PyHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.h)
    file(GLOB PyHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/BatchProvider.h)
    file(GLOB PyHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/PosStreamer.h)
    file(GLOB PyHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/Sample.h)
    file(GLOB PyHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Position.h)



    list(REMOVE_ITEM TSources  ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.cpp)
    list(REMOVE_ITEM THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.h)
    list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.cpp)
    list(REMOVE_ITEM THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/PyHelper.h)
    list(REMOVE_ITEM THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/egdb.h)


endif ()


file(GLOB CHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/*.h)
file(GLOB CSources ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/*.cpp)
list(REMOVE_ITEM CHeaders "${CMAKE_//CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.h")
list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.cpp")
list(REMOVE_ITEM CHeaders "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb.h")
list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/test.cpp")
list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp")
list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/test.cpp)
list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/rescorer.cpp)



add_executable(MainEngine ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources})
if (UNIX)
add_executable(testing ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/test.cpp ${CHeaders} ${CSources})
#add_executable(Simd ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Simd.cpp)
set_target_properties(MainEngine PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
add_executable(perft_check ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Checks/perft_check.cpp ${CHeaders} ${CSources})
endif()

if(WIN32)
  add_library(Dark SHARED ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.h ${CHeaders} ${CSources})
  add_executable(Rescorer ${CMAKE_CURRENT_SOURCE_DIR}/Training/rescorer.cpp ${CHeaders} ${CSources} ${CMAKE_CURRENT_SOURCE_DIR}/Training/Sample.h ${CMAKE_CURRENT_SOURCE_DIR}/Training/Sample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Training/egdb.h)
  set_target_properties(Rescorer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
  target_compile_definitions(Dark PUBLIC "CHECKERBOARD")
  target_compile_definitions(MainEngine PUBLIC "USEDB")
  target_link_libraries(Rescorer ${CMAKE_CURRENT_SOURCE_DIR}/Training/egdb64.lib Threads::Threads)
  target_link_libraries(Dark PRIVATE Threads::Threads)

endif()

if(UNIX)
  #no longer working
  add_library(pyhelper SHARED ${PROTO_SRCS} ${PROTO_HDRS} ${CHeaders} ${CSources} ${PySources} ${PyHeaders})
endif()

if (UNIX)
    add_executable(Train ${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp ${THeaders} ${TSources} ${CHeaders} ${CSources} ${PROTO_SRCS} ${PROTO_HDRS})
    set_target_properties(Train PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
    set_target_properties(perft_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
endif ()
set_target_properties(MainEngine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")

if(UNIX)
target_link_libraries(Train ${Protobuf_LIBRARIES})
target_link_libraries(pyhelper ${Protobuf_LIBRARIES})
target_link_libraries(MainEngine PRIVATE Threads::Threads)
endif()
if(WIN32)
target_link_libraries(MainEngine  ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib)
endif()
