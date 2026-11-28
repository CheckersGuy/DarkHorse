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

# Lazy SMP search threads (Windows targets link winpthread explicitly below)
if (UNIX)
  find_package(Threads REQUIRED)
  link_libraries(Threads::Threads)
endif()


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
list(REMOVE_ITEM CSources "${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/netpack_tool.cpp")
list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/test.cpp)
list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp)
list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/permutation.cpp)
list(REMOVE_ITEM TSources ${CMAKE_CURRENT_SOURCE_DIR}/Training/rescorer.cpp)



add_executable(MainEngine ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources})
# VNNI variant: identical to MainEngine (x86-64-v3 / AVX2) plus AVX-VNNI, so the int8
# dot products of the nets use vpdpbusd (Simd.h). Runs on Intel Core 12th gen+ and
# AMD Zen 5; the binary checks the CPU at startup. Bit-exact with the AVX2 build.
set(VNNI_FLAGS "-mavxvnni")
add_executable(MainEngine-vnni ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources})
target_compile_options(MainEngine-vnni PRIVATE ${VNNI_FLAGS})
target_compile_definitions(MainEngine-vnni PUBLIC "USEDB")
add_executable(generator ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/main.cpp ${CHeaders} ${CSources})
target_compile_definitions(generator PUBLIC "GENERATOR")
if (UNIX)
target_compile_definitions(MainEngine PUBLIC "USEDB")
#add_executable(Simd ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Simd.cpp)
add_executable(perft_check ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/Checks/perft_check.cpp ${CHeaders} ${CSources})
endif()

if(WIN32)
  set(WIN32_STATIC_PTHREAD
    -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive,-Bdynamic
  )
  # CheckerBoard engines: building the target "Dark" produces both DLLs,
  # DarkHorse-<version>-avx2.dll and DarkHorse-<version>-vnni.dll (AVX-VNNI, see
  # MainEngine-vnni). The version is also the one CheckerBoard shows ("name").
  set(DARKHORSE_VERSION "1.1")
  foreach(variant avx2 vnni)
    add_library(Dark-${variant} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/CheckerBoard.h ${CHeaders} ${CSources})
    target_compile_definitions(Dark-${variant} PUBLIC "CHECKERBOARD" "DARKHORSE_VERSION=\"${DARKHORSE_VERSION}\"")
    set_target_properties(Dark-${variant} PROPERTIES PREFIX "" OUTPUT_NAME "DarkHorse-${DARKHORSE_VERSION}-${variant}")
    target_link_libraries(Dark-${variant} ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib ${WIN32_STATIC_PTHREAD})
  endforeach()
  target_compile_options(Dark-vnni PRIVATE ${VNNI_FLAGS})
  add_custom_target(Dark DEPENDS Dark-avx2 Dark-vnni)
  target_compile_definitions(MainEngine PUBLIC "USEDB")

  add_library(RustDll SHARED ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/helper.h ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/helper.cpp ${CHeaders} ${CSources})

  target_link_libraries(MainEngine ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib ${WIN32_STATIC_PTHREAD})
  target_link_libraries(MainEngine-vnni ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib ${WIN32_STATIC_PTHREAD})
  target_link_libraries(RustDll ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/egdb64.lib )
endif()
add_executable(sparse_opt ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/test.cpp ${CHeaders} ${CSources})
target_compile_definitions(sparse_opt PUBLIC "SPARSEOPT")


if (UNIX)
    add_library(RustDll SHARED ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/helper.h ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/helper.cpp ${CHeaders} ${CSources})
    add_executable(Train ${CMAKE_CURRENT_SOURCE_DIR}/Training/main.cpp ${THeaders} ${TSources} ${CHeaders} ${CSources})
    add_executable(Permutation ${CMAKE_CURRENT_SOURCE_DIR}/Training/permutation.cpp ${THeaders} ${TSources} ${CHeaders} ${CSources})
    # packs a .quant net into the compressed .dhn format embedded in the engine (NetPack.h)
    add_executable(NetPack ${CMAKE_CURRENT_SOURCE_DIR}/Checkers/CheckerEngineX/netpack_tool.cpp)
    set_target_properties(NetPack PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
    set_target_properties(Train PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
    set_target_properties(perft_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
    set_target_properties(sparse_opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
endif ()
set_target_properties(MainEngine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
set_target_properties(MainEngine-vnni PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")
set_target_properties(generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")

