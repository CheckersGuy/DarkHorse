# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/103/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/103/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/robin/DarkHorse/Checkers/CheckerEngineX

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/robin/DarkHorse/Checkers/CheckerEngineX

# Include any dependencies generated for this target.
include CMakeFiles/CheckerEngineX.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CheckerEngineX.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CheckerEngineX.dir/flags.make

CMakeFiles/CheckerEngineX.dir/main.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CheckerEngineX.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/main.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/main.cpp

CMakeFiles/CheckerEngineX.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/main.cpp > CMakeFiles/CheckerEngineX.dir/main.cpp.i

CMakeFiles/CheckerEngineX.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/main.cpp -o CMakeFiles/CheckerEngineX.dir/main.cpp.s

CMakeFiles/CheckerEngineX.dir/Board.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/Board.cpp.o: Board.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/CheckerEngineX.dir/Board.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/Board.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Board.cpp

CMakeFiles/CheckerEngineX.dir/Board.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/Board.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Board.cpp > CMakeFiles/CheckerEngineX.dir/Board.cpp.i

CMakeFiles/CheckerEngineX.dir/Board.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/Board.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Board.cpp -o CMakeFiles/CheckerEngineX.dir/Board.cpp.s

CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.o: GameLogic.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/GameLogic.cpp

CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/GameLogic.cpp > CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.i

CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/GameLogic.cpp -o CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.s

CMakeFiles/CheckerEngineX.dir/Transposition.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/Transposition.cpp.o: Transposition.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/CheckerEngineX.dir/Transposition.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/Transposition.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Transposition.cpp

CMakeFiles/CheckerEngineX.dir/Transposition.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/Transposition.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Transposition.cpp > CMakeFiles/CheckerEngineX.dir/Transposition.cpp.i

CMakeFiles/CheckerEngineX.dir/Transposition.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/Transposition.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Transposition.cpp -o CMakeFiles/CheckerEngineX.dir/Transposition.cpp.s

CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.o: MoveListe.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/MoveListe.cpp

CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/MoveListe.cpp > CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.i

CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/MoveListe.cpp -o CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.s

CMakeFiles/CheckerEngineX.dir/Position.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/Position.cpp.o: Position.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/CheckerEngineX.dir/Position.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/Position.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Position.cpp

CMakeFiles/CheckerEngineX.dir/Position.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/Position.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Position.cpp > CMakeFiles/CheckerEngineX.dir/Position.cpp.i

CMakeFiles/CheckerEngineX.dir/Position.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/Position.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Position.cpp -o CMakeFiles/CheckerEngineX.dir/Position.cpp.s

CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.o: MovePicker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/MovePicker.cpp

CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/MovePicker.cpp > CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.i

CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/MovePicker.cpp -o CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.s

CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.o: CBInterface.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/CBInterface.cpp

CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/CBInterface.cpp > CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.i

CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/CBInterface.cpp -o CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.s

CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.o: Zobrist.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Zobrist.cpp

CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Zobrist.cpp > CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.i

CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Zobrist.cpp -o CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.s

CMakeFiles/CheckerEngineX.dir/Line.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/Line.cpp.o: Line.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/CheckerEngineX.dir/Line.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/Line.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Line.cpp

CMakeFiles/CheckerEngineX.dir/Line.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/Line.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Line.cpp > CMakeFiles/CheckerEngineX.dir/Line.cpp.i

CMakeFiles/CheckerEngineX.dir/Line.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/Line.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Line.cpp -o CMakeFiles/CheckerEngineX.dir/Line.cpp.s

CMakeFiles/CheckerEngineX.dir/Thread.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/Thread.cpp.o: Thread.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/CheckerEngineX.dir/Thread.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/Thread.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Thread.cpp

CMakeFiles/CheckerEngineX.dir/Thread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/Thread.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Thread.cpp > CMakeFiles/CheckerEngineX.dir/Thread.cpp.i

CMakeFiles/CheckerEngineX.dir/Thread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/Thread.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Thread.cpp -o CMakeFiles/CheckerEngineX.dir/Thread.cpp.s

CMakeFiles/CheckerEngineX.dir/Move.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/Move.cpp.o: Move.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/CheckerEngineX.dir/Move.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/Move.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Move.cpp

CMakeFiles/CheckerEngineX.dir/Move.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/Move.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Move.cpp > CMakeFiles/CheckerEngineX.dir/Move.cpp.i

CMakeFiles/CheckerEngineX.dir/Move.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/Move.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Move.cpp -o CMakeFiles/CheckerEngineX.dir/Move.cpp.s

CMakeFiles/CheckerEngineX.dir/Bits.cpp.o: CMakeFiles/CheckerEngineX.dir/flags.make
CMakeFiles/CheckerEngineX.dir/Bits.cpp.o: Bits.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/CheckerEngineX.dir/Bits.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CheckerEngineX.dir/Bits.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Bits.cpp

CMakeFiles/CheckerEngineX.dir/Bits.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CheckerEngineX.dir/Bits.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Bits.cpp > CMakeFiles/CheckerEngineX.dir/Bits.cpp.i

CMakeFiles/CheckerEngineX.dir/Bits.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CheckerEngineX.dir/Bits.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Bits.cpp -o CMakeFiles/CheckerEngineX.dir/Bits.cpp.s

# Object files for target CheckerEngineX
CheckerEngineX_OBJECTS = \
"CMakeFiles/CheckerEngineX.dir/main.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/Board.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/Transposition.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/Position.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/Line.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/Thread.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/Move.cpp.o" \
"CMakeFiles/CheckerEngineX.dir/Bits.cpp.o"

# External object files for target CheckerEngineX
CheckerEngineX_EXTERNAL_OBJECTS =

cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/main.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/Board.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/GameLogic.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/Transposition.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/MoveListe.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/Position.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/MovePicker.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/CBInterface.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/Zobrist.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/Line.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/Thread.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/Move.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/Bits.cpp.o
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/build.make
cmake-build-debug/CheckerEngineX: CMakeFiles/CheckerEngineX.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX executable cmake-build-debug/CheckerEngineX"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CheckerEngineX.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CheckerEngineX.dir/build: cmake-build-debug/CheckerEngineX

.PHONY : CMakeFiles/CheckerEngineX.dir/build

CMakeFiles/CheckerEngineX.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CheckerEngineX.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CheckerEngineX.dir/clean

CMakeFiles/CheckerEngineX.dir/depend:
	cd /home/robin/DarkHorse/Checkers/CheckerEngineX && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles/CheckerEngineX.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CheckerEngineX.dir/depend

