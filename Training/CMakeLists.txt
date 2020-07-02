cmake_minimum_required(VERSION 3.6)
project(Training)

set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(TRAIN)
set(CMAKE_CXX_FLAGS "-O2 -DNDEBUG -march=native -pthread  -flto -ffat-lto-objects -fuse-linker-plugin")
set(CMAKE_CXX_STANDARD_LIBRARIES -ldl )
find_package(Protobuf REQUIRED)
include_directories(../Checkers/CheckerEngineX)


add_executable(Training main.cpp Trainer.cpp Trainer.h Match.cpp Match.h
        Utilities.cpp Utilities.h
        ../Checkers/CheckerEngineX/types.h
        ../Checkers/CheckerEngineX/Position.cpp ../Checkers/CheckerEngineX/Position.h
        ../Checkers/CheckerEngineX/Move.h  ../Checkers/CheckerEngineX/Move.cpp
        ../Checkers/CheckerEngineX/Board.cpp ../Checkers/CheckerEngineX/Board.h
        ../Checkers/CheckerEngineX/Zobrist.cpp ../Checkers/CheckerEngineX/Zobrist.h
        ../Checkers/CheckerEngineX/GameLogic.h ../Checkers/CheckerEngineX/GameLogic.cpp
        ../Checkers/CheckerEngineX/Transposition.cpp ../Checkers/CheckerEngineX/Transposition.h
        ../Checkers/CheckerEngineX/MovePicker.h ../Checkers/CheckerEngineX/MovePicker.cpp
        ../Checkers/CheckerEngineX/MoveListe.h ../Checkers/CheckerEngineX/MoveListe.cpp
        ../Checkers/CheckerEngineX/MGenerator.h
        ../Checkers/CheckerEngineX/CBInterface.h ../Checkers/CheckerEngineX/Line.h
        ../Checkers/CheckerEngineX/CBInterface.cpp ../Checkers/CheckerEngineX/Line.cpp
        ../Checkers/CheckerEngineX/SMPLock.h
        ../Checkers/CheckerEngineX/Bits.cpp ../Checkers/CheckerEngineX/Bits.h
        proto/Training.pb.h proto/Training.pb.cc)

target_link_libraries( Training ${PROTOBUF_LIBRARY})

