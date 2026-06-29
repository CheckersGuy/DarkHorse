cmake_minimum_required(VERSION 3.9)
project(reading)
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_BUILD_TYPE "Release")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  add_compile_options("-march=x86-64-v3")
  add_compile_options("-O3")

add_link_options(
    -static-libstdc++
    -static-libgcc
)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX")


if (UNIX)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Training)
    file(GLOB THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.h)
    file(GLOB TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Training/*.cc)
    file(GLOB PySources ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Position.cpp)
    file(GLOB PyHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Position.h)
    list(REMOVE_ITEM TSources  ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.cpp)
    list(REMOVE_ITEM THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.h)
    list(REMOVE_ITEM THeaders ${CMAKE_CURRENT_SOURCE_DIR}/Training/egdb.h)
endif ()


file(GLOB CHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/*.h)
file(GLOB CSources ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/*.cpp)

if (UNIX)
  list(REMOVE_ITEM CHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Endgame.h)
  list(REMOVE_ITEM CSources ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Endgame.cpp)
  list(REMOVE_ITEM CHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/helper.h)
  list(REMOVE_ITEM CSources ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/helper.cpp)
  list(REMOVE_ITEM CHeaders ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/registry.h)
  list(REMOVE_ITEM CSources ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/registry.cpp)
endif()


list(REMOVE_ITEM CHeaders "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.h")
list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.cpp")

if(UNIX)
list(REMOVE_ITEM CHeaders "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb.h")
list(REMOVE_ITEM CHeaders "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egd64.lib")
endif()

list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/test.cpp")
list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp")
list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/test.cpp)
list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp)
list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/permutation.cpp)
list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/rescorer.cpp)



add_executable(MainEngine ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources})
add_executable(generator ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources})
target_compile_definitions(generator PUBLIC "GENERATOR")
if (UNIX)
target_compile_definitions(MainEngine PUBLIC "USEDB")
#add_executable(Simd ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Simd.cpp)
add_executable(perft_check ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Checks/perft_check.cpp ${CHeaders} ${CSources})
endif()

if(WIN32)
  add_library(Dark SHARED ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.h ${CHeaders} ${CSources})
  target_compile_definitions(Dark PUBLIC "CHECKERBOARD")
  target_compile_definitions(MainEngine PUBLIC "USEDB")

  add_library(RustDll SHARED ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/helper.h ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/helper.cpp ${CHeaders} ${CSources})
    set(WIN32_STATIC_PTHREAD
      -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive,-Bdynamic
  )

  target_link_libraries(MainEngine ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib ${WIN32_STATIC_PTHREAD})
  target_link_libraries(Dark ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib ${WIN32_STATIC_PTHREAD})
  target_link_libraries(RustDll ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib )
endif()
add_executable(sparse_opt ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/test.cpp ${CHeaders} ${CSources})
target_compile_definitions(sparse_opt PUBLIC "SPARSEOPT")


if (UNIX)
    add_library(RustDll SHARED ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/helper.h ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/helper.cpp ${CHeaders} ${CSources})
    add_executable(Train ${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp ${THeaders} ${TSources} ${CHeaders} ${CSources})
    add_executable(Permutation ${CMAKE_CURRENT_SOURCE_DIR}/Training/permutation.cpp ${THeaders} ${TSources} ${CHeaders} ${CSources})
    set_target_properties(Train PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
    set_target_properties(perft_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
    set_target_properties(sparse_opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
endif ()
set_target_properties(MainEngine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
set_target_properties(generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
if(UNIX)
target_link_libraries(MainEngine PRIVATE Threads::Threads)
endif()

