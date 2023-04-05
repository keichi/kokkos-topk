cmake_minimum_required(VERSION 3.16)

project(kokkos-topok CXX)

add_subdirectory(kokkos EXCLUDE_FROM_ALL)

add_executable(topk main.cpp)
target_link_libraries(topk PRIVATE Kokkos::kokkos)
