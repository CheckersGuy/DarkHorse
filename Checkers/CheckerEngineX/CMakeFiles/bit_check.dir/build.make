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
CMAKE_COMMAND = /snap/clion/97/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/97/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/robin/DarkHorse/Checkers/CheckerEngineX

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/robin/DarkHorse/Checkers/CheckerEngineX

# Include any dependencies generated for this target.
include CMakeFiles/bit_check.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bit_check.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bit_check.dir/flags.make

CMakeFiles/bit_check.dir/Bits.cpp.o: CMakeFiles/bit_check.dir/flags.make
CMakeFiles/bit_check.dir/Bits.cpp.o: Bits.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bit_check.dir/Bits.cpp.o"
	/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bit_check.dir/Bits.cpp.o -c /home/robin/DarkHorse/Checkers/CheckerEngineX/Bits.cpp

CMakeFiles/bit_check.dir/Bits.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bit_check.dir/Bits.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robin/DarkHorse/Checkers/CheckerEngineX/Bits.cpp > CMakeFiles/bit_check.dir/Bits.cpp.i

CMakeFiles/bit_check.dir/Bits.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bit_check.dir/Bits.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robin/DarkHorse/Checkers/CheckerEngineX/Bits.cpp -o CMakeFiles/bit_check.dir/Bits.cpp.s

# Object files for target bit_check
bit_check_OBJECTS = \
"CMakeFiles/bit_check.dir/Bits.cpp.o"

# External object files for target bit_check
bit_check_EXTERNAL_OBJECTS =

cmake-build-debug/bit_check: CMakeFiles/bit_check.dir/Bits.cpp.o
cmake-build-debug/bit_check: CMakeFiles/bit_check.dir/build.make
cmake-build-debug/bit_check: CMakeFiles/bit_check.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cmake-build-debug/bit_check"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bit_check.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bit_check.dir/build: cmake-build-debug/bit_check

.PHONY : CMakeFiles/bit_check.dir/build

CMakeFiles/bit_check.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bit_check.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bit_check.dir/clean

CMakeFiles/bit_check.dir/depend:
	cd /home/robin/DarkHorse/Checkers/CheckerEngineX && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX /home/robin/DarkHorse/Checkers/CheckerEngineX/CMakeFiles/bit_check.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bit_check.dir/depend
