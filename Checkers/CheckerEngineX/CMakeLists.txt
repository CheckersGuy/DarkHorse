cmake_minimum_required(VERSION 3.6)
project(CheckerEngineX)
set(CMAKE_CXX_STANDARD 14)

#find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "-O3  -DNDEBUG -march=native -fpeel-loops  -fomit-frame-pointer  -ftree-loop-if-convert  -flto -ffat-lto-objects -fuse-linker-plugin ")
set(SOURCE_FILES main.cpp Board.cpp Board.h BoardFactory.cpp BoardFactory.h Move.h MGenerator.cpp MGenerator.h GameEvaluation.cpp GameEvaluation.h GameLogic.cpp GameLogic.h Transposition.cpp Transposition.h types.h MoveListe.cpp MoveListe.h  Position.cpp Position.h MovePicker.cpp MovePicker.h CBInterface.cpp CBInterface.h Zobrist.h Zobrist.cpp Weights.h Line.cpp Line.h)

add_executable(CheckerEngineX  ${SOURCE_FILES})
#target_link_libraries (CheckerEngineX ${CMAKE_THREAD_LIBS_INIT})