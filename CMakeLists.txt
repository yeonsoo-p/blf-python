cmake_minimum_required(VERSION 3.15)
project(blf_python C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Development NumPy)

# NumPy include directory
set(NUMPY_INCLUDE_DIR "${Python3_NumPy_INCLUDE_DIRS}")

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_dbc/src
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)

# Add the Python extension module
Python3_add_library(blf_python MODULE
    src/blf_module.cpp
)

# Link against both binlog and vector_dbc libraries
# Use -Wl,-Bstatic to force static linking of runtime libraries
target_link_libraries(blf_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/binlog.dll.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libVector_DBC.dll.a
    -Wl,-Bstatic
    -lstdc++
    -lwinpthread
    -Wl,-Bdynamic
    -static-libgcc
)

# Set compiler flags (disable cast-function-type warning for Python C API compatibility)
target_compile_options(blf_python PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra -Wno-cast-function-type)

# Set output directory
set_target_properties(blf_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/blf_python
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/blf_python
)

# Remove the 'lib' prefix on Windows if present
set_target_properties(blf_python PROPERTIES PREFIX "")

# Ensure .pyd extension on Windows
if(WIN32)
    set_target_properties(blf_python PROPERTIES SUFFIX ".pyd")
endif()

# Add read_blf executable
add_executable(read_blf
    test/read_blf.c
)

# Link binlog import library for read_blf executable
target_link_libraries(read_blf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/binlog.dll.a
)

# Add decode_can executable
add_executable(decode_can
    test/decode_can.cpp
)

# Link Vector_DBC library for decode_can executable
target_link_libraries(decode_can PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libVector_DBC.dll.a
)


# Copy required DLLs to blf_python directory for Python module to load
add_custom_command(TARGET blf_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/binlog.dll
        ${CMAKE_CURRENT_SOURCE_DIR}/blf_python/binlog.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libVector_DBC.dll
        ${CMAKE_CURRENT_SOURCE_DIR}/blf_python/libVector_DBC.dll
    COMMENT "Copying DLLs to blf_python directory"
)
