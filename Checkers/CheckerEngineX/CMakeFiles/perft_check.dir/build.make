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
CMAKE_COMMAND = /snap/clion/98/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/98/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/robin/DarkHorse/Checkers/CheckerEngineX

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/robin/DarkHorse/Checkers/CheckerEngineX

# Include any dependencies generated for this target.
include CMakeFiles/perft_check.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/perft_check.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/perft_check.dir/flags.make

CMakeFiles/perft_check.dir/Checks/perft_check.cpp.o: CMakeFiles/perft_check.dir/flags.make
CMakeFiles/perft_check.dir/Checks/perft_check.cpp.o: Checks/perft_check.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/perft_check.dir/Checks/perft_check.cpp.o"
	/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/perft_check.dir/Checks/perft_check.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Checks/perft_check.cpp

CMakeFiles/perft_check.dir/Checks/perft_check.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perft_check.dir/Checks/perft_check.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Checks/perft_check.cpp > CMakeFiles/perft_check.dir/Checks/perft_check.cpp.i

CMakeFiles/perft_check.dir/Checks/perft_check.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perft_check.dir/Checks/perft_check.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Checks/perft_check.cpp -o CMakeFiles/perft_check.dir/Checks/perft_check.cpp.s

CMakeFiles/perft_check.dir/Perft.cpp.o: CMakeFiles/perft_check.dir/flags.make
CMakeFiles/perft_check.dir/Perft.cpp.o: Perft.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/perft_check.dir/Perft.cpp.o"
	/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/perft_check.dir/Perft.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Perft.cpp

CMakeFiles/perft_check.dir/Perft.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perft_check.dir/Perft.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Perft.cpp > CMakeFiles/perft_check.dir/Perft.cpp.i

CMakeFiles/perft_check.dir/Perft.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perft_check.dir/Perft.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Perft.cpp -o CMakeFiles/perft_check.dir/Perft.cpp.s

CMakeFiles/perft_check.dir/Zobrist.cpp.o: CMakeFiles/perft_check.dir/flags.make
CMakeFiles/perft_check.dir/Zobrist.cpp.o: Zobrist.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/perft_check.dir/Zobrist.cpp.o"
	/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/perft_check.dir/Zobrist.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Zobrist.cpp

CMakeFiles/perft_check.dir/Zobrist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perft_check.dir/Zobrist.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Zobrist.cpp > CMakeFiles/perft_check.dir/Zobrist.cpp.i

CMakeFiles/perft_check.dir/Zobrist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perft_check.dir/Zobrist.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Zobrist.cpp -o CMakeFiles/perft_check.dir/Zobrist.cpp.s

CMakeFiles/perft_check.dir/Board.cpp.o: CMakeFiles/perft_check.dir/flags.make
CMakeFiles/perft_check.dir/Board.cpp.o: Board.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/perft_check.dir/Board.cpp.o"
	/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/perft_check.dir/Board.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Board.cpp

CMakeFiles/perft_check.dir/Board.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perft_check.dir/Board.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Board.cpp > CMakeFiles/perft_check.dir/Board.cpp.i

CMakeFiles/perft_check.dir/Board.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perft_check.dir/Board.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Board.cpp -o CMakeFiles/perft_check.dir/Board.cpp.s

CMakeFiles/perft_check.dir/Position.cpp.o: CMakeFiles/perft_check.dir/flags.make
CMakeFiles/perft_check.dir/Position.cpp.o: Position.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/perft_check.dir/Position.cpp.o"
	/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/perft_check.dir/Position.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Position.cpp

CMakeFiles/perft_check.dir/Position.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perft_check.dir/Position.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Position.cpp > CMakeFiles/perft_check.dir/Position.cpp.i

CMakeFiles/perft_check.dir/Position.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perft_check.dir/Position.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Position.cpp -o CMakeFiles/perft_check.dir/Position.cpp.s

CMakeFiles/perft_check.dir/Bits.cpp.o: CMakeFiles/perft_check.dir/flags.make
CMakeFiles/perft_check.dir/Bits.cpp.o: Bits.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/perft_check.dir/Bits.cpp.o"
	/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/perft_check.dir/Bits.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Bits.cpp

CMakeFiles/perft_check.dir/Bits.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perft_check.dir/Bits.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Bits.cpp > CMakeFiles/perft_check.dir/Bits.cpp.i

CMakeFiles/perft_check.dir/Bits.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perft_check.dir/Bits.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Bits.cpp -o CMakeFiles/perft_check.dir/Bits.cpp.s

CMakeFiles/perft_check.dir/Move.cpp.o: CMakeFiles/perft_check.dir/flags.make
CMakeFiles/perft_check.dir/Move.cpp.o: Move.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/perft_check.dir/Move.cpp.o"
	/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/perft_check.dir/Move.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Move.cpp

CMakeFiles/perft_check.dir/Move.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perft_check.dir/Move.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Move.cpp > CMakeFiles/perft_check.dir/Move.cpp.i

CMakeFiles/perft_check.dir/Move.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perft_check.dir/Move.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Move.cpp -o CMakeFiles/perft_check.dir/Move.cpp.s

# Object files for target perft_check
perft_check_OBJECTS = \
"CMakeFiles/perft_check.dir/Checks/perft_check.cpp.o" \
"CMakeFiles/perft_check.dir/Perft.cpp.o" \
"CMakeFiles/perft_check.dir/Zobrist.cpp.o" \
"CMakeFiles/perft_check.dir/Board.cpp.o" \
"CMakeFiles/perft_check.dir/Position.cpp.o" \
"CMakeFiles/perft_check.dir/Bits.cpp.o" \
"CMakeFiles/perft_check.dir/Move.cpp.o"

# External object files for target perft_check
perft_check_EXTERNAL_OBJECTS =

cmake-build-debug/perft_check: CMakeFiles/perft_check.dir/Checks/perft_check.cpp.o
cmake-build-debug/perft_check: CMakeFiles/perft_check.dir/Perft.cpp.o
cmake-build-debug/perft_check: CMakeFiles/perft_check.dir/Zobrist.cpp.o
cmake-build-debug/perft_check: CMakeFiles/perft_check.dir/Board.cpp.o
cmake-build-debug/perft_check: CMakeFiles/perft_check.dir/Position.cpp.o
cmake-build-debug/perft_check: CMakeFiles/perft_check.dir/Bits.cpp.o
cmake-build-debug/perft_check: CMakeFiles/perft_check.dir/Move.cpp.o
cmake-build-debug/perft_check: CMakeFiles/perft_check.dir/build.make
cmake-build-debug/perft_check: CMakeFiles/perft_check.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable cmake-build-debug/perft_check"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/perft_check.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/perft_check.dir/build: cmake-build-debug/perft_check

.PHONY : CMakeFiles/perft_check.dir/build

CMakeFiles/perft_check.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/perft_check.dir/cmake_clean.cmake
.PHONY : CMakeFiles/perft_check.dir/clean

CMakeFiles/perft_check.dir/depend:
	cd /home/robin/DarkHorse/Checkers/CheckerEngineX && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles/perft_check.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/perft_check.dir/depend

