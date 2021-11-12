cmake_minimum_required(VERSION 3.7)
project(cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_OSX_DEPLOYMENT_TARGET 11.6)

#set(CMAKE_VERBOSE_MAKEFILE 1)

find_package(MPI)
link_directories(/usr/local/lib)

add_executable(test_Scalapack test_Scalapack.cpp Scalapack.cpp)
target_link_libraries(test_Scalapack PRIVATE MPI::MPI_CXX)
target_link_libraries(test_Scalapack PRIVATE scalapack)

