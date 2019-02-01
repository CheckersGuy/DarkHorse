cmake_minimum_required(VERSION 3.10)
project(Training)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -DTRAIN -DNDEBUG -march=native -pthread  -flto -ffat-lto-objects -fuse-linker-plugin")
set(CMAKE_CXX_STANDARD_LIBRARIES -ldl )
set(Boost_USE_STATIC_LIBS   ON)

find_package( Boost REQUIRED COMPONENTS program_options system iostreams filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )




include_directories(../Checkers/CheckerEngineX)


add_executable(Training main.cpp Trainer.cpp Trainer.h Training.h Training.cpp Engine.cpp Engine.h Match.cpp Match.h Generator.cpp Generator.h
       SimplePool.h SimplePool.cpp Utilities.cpp Utilities.h
        ../Checkers/CheckerEngineX/types.h
        ../Checkers/CheckerEngineX/Position.cpp ../Checkers/CheckerEngineX/Position.h
        ../Checkers/CheckerEngineX/Move.h
        ../Checkers/CheckerEngineX/Board.cpp ../Checkers/CheckerEngineX/Board.h
        ../Checkers/CheckerEngineX/BoardFactory.cpp ../Checkers/CheckerEngineX/BoardFactory.h
        ../Checkers/CheckerEngineX/Zobrist.cpp ../Checkers/CheckerEngineX/Zobrist.h
        ../Checkers/CheckerEngineX/GameLogic.h ../Checkers/CheckerEngineX/GameLogic.cpp
        ../Checkers/CheckerEngineX/Transposition.cpp ../Checkers/CheckerEngineX/Transposition.h
        ../Checkers/CheckerEngineX/BoardFactory.cpp ../Checkers/CheckerEngineX/BoardFactory.h
        ../Checkers/CheckerEngineX/MovePicker.h ../Checkers/CheckerEngineX/MovePicker.cpp
        ../Checkers/CheckerEngineX/MoveListe.h ../Checkers/CheckerEngineX/MoveListe.cpp
        ../Checkers/CheckerEngineX/MGenerator.h
        ../Checkers/CheckerEngineX/CBInterface.h ../Checkers/CheckerEngineX/Line.h
        ../Checkers/CheckerEngineX/CBInterface.cpp ../Checkers/CheckerEngineX/Line.cpp
        ../Checkers/CheckerEngineX/SMPLock.h Cache.h)

target_link_libraries( Training ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${Boost_SYSTEM_LIBRARY} ${Boost_IOSTREAMS_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} )

