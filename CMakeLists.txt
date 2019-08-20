cmake_minimum_required(VERSION 3.6)

set(CMAKE_CXX_STANDARD 17)

# `function` confuses FetchContent, sees empty CMAKE_CURRENT_LIST_DIR
macro(fetch_and_include name)
    message(STATUS "Fetching ${name}")

    set(fetch_and_include_local_path "${CMAKE_BINARY_DIR}/fetch_and_include/${name}")
    if(NOT EXISTS ${fetch_and_include_local_path})
      file(DOWNLOAD
          "https://raw.githubusercontent.com/boostorg/mincmake/develop/${name}"
          "${CMAKE_BINARY_DIR}/fetch_and_include/${name}"
      )
    endif()
    include("${CMAKE_BINARY_DIR}/fetch_and_include/${name}")
endmacro()

fetch_and_include(cmake/boost_fetch.cmake)

option(BENCHMARK_ENABLE_TESTING "" OFF)
set(CMAKE_BUILD_TYPE Release) # to build google-benchmark in release mode
boost_fetch(google/benchmark)

include_directories(extern/eigen)

macro(add_benchmark name)
  add_executable(${name} "${name}.cpp")
  target_compile_options(${name} PRIVATE
    -DNDEBUG -O3 -march=native ${BENCHMARK_FLAGS} -funsafe-math-optimizations)
  target_link_libraries(${name} PRIVATE benchmark_main)
endmacro()

add_benchmark(span_demo)
