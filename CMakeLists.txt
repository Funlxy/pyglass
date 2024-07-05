cmake_minimum_required (VERSION 3.1)
project(glass LANGUAGES CXX)

# set header file path for glass
add_library(glass INTERFACE)
target_include_directories(glass INTERFACE .) 
# Boost
FIND_PACKAGE(Boost COMPONENTS program_options system REQUIRED)
# IF (Boost_FOUND)
#     INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
#     message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
# ENDIF()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CXX_STANDARD 20)
# SET(CMAKE_CXX_FLAGS  "-Wall -Wextra -O3 -lrt -std=c++20 -march=native -fpic -fopenmp -ftree-vectorize" )

# common flags
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -lrt -std=c++20 -fpic -fopenmp -mtune=native" )

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -DDEBUG ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG -march=native -ftree-vectorize")

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release CACHE STRING "Build types: Release Debug" FORCE)
endif()
message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_executable(main examples/main.cc)
target_link_libraries(main glass Boost::program_options)

# add_executable(server examples/server.cpp)
# target_link_libraries(server glass Boost::system)
