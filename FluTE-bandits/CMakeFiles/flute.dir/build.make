# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fiens/SCHOOL/FluTE-bandits

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fiens/SCHOOL/FluTE-bandits

# Include any dependencies generated for this target.
include CMakeFiles/flute.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/flute.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/flute.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/flute.dir/flags.make

CMakeFiles/flute.dir/src/flute.cpp.o: CMakeFiles/flute.dir/flags.make
CMakeFiles/flute.dir/src/flute.cpp.o: src/flute.cpp
CMakeFiles/flute.dir/src/flute.cpp.o: CMakeFiles/flute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fiens/SCHOOL/FluTE-bandits/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/flute.dir/src/flute.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/flute.dir/src/flute.cpp.o -MF CMakeFiles/flute.dir/src/flute.cpp.o.d -o CMakeFiles/flute.dir/src/flute.cpp.o -c /home/fiens/SCHOOL/FluTE-bandits/src/flute.cpp

CMakeFiles/flute.dir/src/flute.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flute.dir/src/flute.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fiens/SCHOOL/FluTE-bandits/src/flute.cpp > CMakeFiles/flute.dir/src/flute.cpp.i

CMakeFiles/flute.dir/src/flute.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flute.dir/src/flute.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fiens/SCHOOL/FluTE-bandits/src/flute.cpp -o CMakeFiles/flute.dir/src/flute.cpp.s

CMakeFiles/flute.dir/src/epimodel.cpp.o: CMakeFiles/flute.dir/flags.make
CMakeFiles/flute.dir/src/epimodel.cpp.o: src/epimodel.cpp
CMakeFiles/flute.dir/src/epimodel.cpp.o: CMakeFiles/flute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fiens/SCHOOL/FluTE-bandits/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/flute.dir/src/epimodel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/flute.dir/src/epimodel.cpp.o -MF CMakeFiles/flute.dir/src/epimodel.cpp.o.d -o CMakeFiles/flute.dir/src/epimodel.cpp.o -c /home/fiens/SCHOOL/FluTE-bandits/src/epimodel.cpp

CMakeFiles/flute.dir/src/epimodel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flute.dir/src/epimodel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fiens/SCHOOL/FluTE-bandits/src/epimodel.cpp > CMakeFiles/flute.dir/src/epimodel.cpp.i

CMakeFiles/flute.dir/src/epimodel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flute.dir/src/epimodel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fiens/SCHOOL/FluTE-bandits/src/epimodel.cpp -o CMakeFiles/flute.dir/src/epimodel.cpp.s

CMakeFiles/flute.dir/src/params.cpp.o: CMakeFiles/flute.dir/flags.make
CMakeFiles/flute.dir/src/params.cpp.o: src/params.cpp
CMakeFiles/flute.dir/src/params.cpp.o: CMakeFiles/flute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fiens/SCHOOL/FluTE-bandits/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/flute.dir/src/params.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/flute.dir/src/params.cpp.o -MF CMakeFiles/flute.dir/src/params.cpp.o.d -o CMakeFiles/flute.dir/src/params.cpp.o -c /home/fiens/SCHOOL/FluTE-bandits/src/params.cpp

CMakeFiles/flute.dir/src/params.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flute.dir/src/params.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fiens/SCHOOL/FluTE-bandits/src/params.cpp > CMakeFiles/flute.dir/src/params.cpp.i

CMakeFiles/flute.dir/src/params.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flute.dir/src/params.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fiens/SCHOOL/FluTE-bandits/src/params.cpp -o CMakeFiles/flute.dir/src/params.cpp.s

CMakeFiles/flute.dir/src/epimodelparameters.cpp.o: CMakeFiles/flute.dir/flags.make
CMakeFiles/flute.dir/src/epimodelparameters.cpp.o: src/epimodelparameters.cpp
CMakeFiles/flute.dir/src/epimodelparameters.cpp.o: CMakeFiles/flute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fiens/SCHOOL/FluTE-bandits/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/flute.dir/src/epimodelparameters.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/flute.dir/src/epimodelparameters.cpp.o -MF CMakeFiles/flute.dir/src/epimodelparameters.cpp.o.d -o CMakeFiles/flute.dir/src/epimodelparameters.cpp.o -c /home/fiens/SCHOOL/FluTE-bandits/src/epimodelparameters.cpp

CMakeFiles/flute.dir/src/epimodelparameters.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flute.dir/src/epimodelparameters.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fiens/SCHOOL/FluTE-bandits/src/epimodelparameters.cpp > CMakeFiles/flute.dir/src/epimodelparameters.cpp.i

CMakeFiles/flute.dir/src/epimodelparameters.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flute.dir/src/epimodelparameters.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fiens/SCHOOL/FluTE-bandits/src/epimodelparameters.cpp -o CMakeFiles/flute.dir/src/epimodelparameters.cpp.s

CMakeFiles/flute.dir/src/sfmt/dSFMT.c.o: CMakeFiles/flute.dir/flags.make
CMakeFiles/flute.dir/src/sfmt/dSFMT.c.o: src/sfmt/dSFMT.c
CMakeFiles/flute.dir/src/sfmt/dSFMT.c.o: CMakeFiles/flute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fiens/SCHOOL/FluTE-bandits/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/flute.dir/src/sfmt/dSFMT.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -std=c99 --param max-inline-insns-single=1800 -fno-strict-aliasing -Wmissing-prototypes -DNDEBUG -MD -MT CMakeFiles/flute.dir/src/sfmt/dSFMT.c.o -MF CMakeFiles/flute.dir/src/sfmt/dSFMT.c.o.d -o CMakeFiles/flute.dir/src/sfmt/dSFMT.c.o -c /home/fiens/SCHOOL/FluTE-bandits/src/sfmt/dSFMT.c

CMakeFiles/flute.dir/src/sfmt/dSFMT.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/flute.dir/src/sfmt/dSFMT.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -std=c99 --param max-inline-insns-single=1800 -fno-strict-aliasing -Wmissing-prototypes -DNDEBUG -E /home/fiens/SCHOOL/FluTE-bandits/src/sfmt/dSFMT.c > CMakeFiles/flute.dir/src/sfmt/dSFMT.c.i

CMakeFiles/flute.dir/src/sfmt/dSFMT.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/flute.dir/src/sfmt/dSFMT.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -std=c99 --param max-inline-insns-single=1800 -fno-strict-aliasing -Wmissing-prototypes -DNDEBUG -S /home/fiens/SCHOOL/FluTE-bandits/src/sfmt/dSFMT.c -o CMakeFiles/flute.dir/src/sfmt/dSFMT.c.s

CMakeFiles/flute.dir/src/bnldev.c.o: CMakeFiles/flute.dir/flags.make
CMakeFiles/flute.dir/src/bnldev.c.o: src/bnldev.c
CMakeFiles/flute.dir/src/bnldev.c.o: CMakeFiles/flute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fiens/SCHOOL/FluTE-bandits/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/flute.dir/src/bnldev.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/flute.dir/src/bnldev.c.o -MF CMakeFiles/flute.dir/src/bnldev.c.o.d -o CMakeFiles/flute.dir/src/bnldev.c.o -c /home/fiens/SCHOOL/FluTE-bandits/src/bnldev.c

CMakeFiles/flute.dir/src/bnldev.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/flute.dir/src/bnldev.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/fiens/SCHOOL/FluTE-bandits/src/bnldev.c > CMakeFiles/flute.dir/src/bnldev.c.i

CMakeFiles/flute.dir/src/bnldev.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/flute.dir/src/bnldev.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/fiens/SCHOOL/FluTE-bandits/src/bnldev.c -o CMakeFiles/flute.dir/src/bnldev.c.s

# Object files for target flute
flute_OBJECTS = \
"CMakeFiles/flute.dir/src/flute.cpp.o" \
"CMakeFiles/flute.dir/src/epimodel.cpp.o" \
"CMakeFiles/flute.dir/src/params.cpp.o" \
"CMakeFiles/flute.dir/src/epimodelparameters.cpp.o" \
"CMakeFiles/flute.dir/src/sfmt/dSFMT.c.o" \
"CMakeFiles/flute.dir/src/bnldev.c.o"

# External object files for target flute
flute_EXTERNAL_OBJECTS =

flute: CMakeFiles/flute.dir/src/flute.cpp.o
flute: CMakeFiles/flute.dir/src/epimodel.cpp.o
flute: CMakeFiles/flute.dir/src/params.cpp.o
flute: CMakeFiles/flute.dir/src/epimodelparameters.cpp.o
flute: CMakeFiles/flute.dir/src/sfmt/dSFMT.c.o
flute: CMakeFiles/flute.dir/src/bnldev.c.o
flute: CMakeFiles/flute.dir/build.make
flute: CMakeFiles/flute.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fiens/SCHOOL/FluTE-bandits/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable flute"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/flute.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/flute.dir/build: flute
.PHONY : CMakeFiles/flute.dir/build

CMakeFiles/flute.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/flute.dir/cmake_clean.cmake
.PHONY : CMakeFiles/flute.dir/clean

CMakeFiles/flute.dir/depend:
	cd /home/fiens/SCHOOL/FluTE-bandits && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fiens/SCHOOL/FluTE-bandits /home/fiens/SCHOOL/FluTE-bandits /home/fiens/SCHOOL/FluTE-bandits /home/fiens/SCHOOL/FluTE-bandits /home/fiens/SCHOOL/FluTE-bandits/CMakeFiles/flute.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/flute.dir/depend
