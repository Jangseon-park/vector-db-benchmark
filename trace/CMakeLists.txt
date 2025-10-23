cmake_minimum_required(VERSION 3.10)
project(mem_tracer_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mem_tracer_test main.cpp mem_trace.cpp)

target_compile_options(mem_tracer_test PRIVATE -Wall -Wextra -g)
