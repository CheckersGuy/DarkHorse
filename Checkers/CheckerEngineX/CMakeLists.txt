cmake_minimum_required(VERSION 3.6)
project(CheckerEngineX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-O3 -pthread  -DNDEBUG -march=native   -flto -ffat-lto-objects -fuse-linker-plugin ")
set(SOURCE_FILES main.cpp Perft.cpp Perft.h Board.cpp Board.h BoardFactory.cpp BoardFactory.h Move.h MGenerator.h GameLogic.cpp GameLogic.h Transposition.cpp Transposition.h types.h MoveListe.cpp MoveListe.h  Position.cpp Position.h MovePicker.cpp MovePicker.h CBInterface.cpp CBInterface.h Zobrist.h Zobrist.cpp Weights.h Line.cpp Line.h SMPLock.h Thread.cpp Thread.h)
add_executable(CheckerEngineX  ${SOURCE_FILES})
