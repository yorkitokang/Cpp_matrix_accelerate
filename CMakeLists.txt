cmake_minimum_required(VERSION 3.23)

project(project4 C)

aux_source_directory(./src DIR_SRCS)

set(CMAKE_C_STANDARD 11)

include_directories(./include)

add_definitions(-fopenmp -lgomp -DWITH_AVX2 -mavx -O3)

add_executable(project4 ${DIR_SRCS})

target_link_libraries(project4 cblas)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()