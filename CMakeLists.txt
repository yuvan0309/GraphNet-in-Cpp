cmake_minimum_required(VERSION 3.14)
project(graphnet-cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch package
find_package(Torch REQUIRED)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Define source files
set(SOURCES
    main.cpp
    graph.cpp
    graphnet.cpp
)

# Define header files
set(HEADERS
    graph.h
    graphnet.h
)

# Create executable
add_executable(graphnet ${SOURCES} ${HEADERS})

# Link LibTorch
target_link_libraries(graphnet ${TORCH_LIBRARIES})

# Set warning level
if(MSVC)
  target_compile_options(graphnet PRIVATE /W4)
else()
  target_compile_options(graphnet PRIVATE -Wall -Wextra)
endif()

# Set output directory
set_target_properties(graphnet PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Print LibTorch version and device info
message(STATUS "LibTorch version: ${TORCH_VERSION}")
message(STATUS "LibTorch include directories: ${TORCH_INCLUDE_DIRS}")