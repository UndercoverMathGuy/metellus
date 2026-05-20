cmake_minimum_required(VERSION 3.18)
project(triton_metal LANGUAGES CXX OBJCXX)

if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Python interpreter" FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(nanobind_DIR "${NANOBIND_CMAKE_DIR}" CACHE PATH "Path to nanobind CMake package configuration")

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(metal_backend src/metal_backend.mm)

target_compile_features(metal_backend PRIVATE cxx_std_17)
target_link_libraries(metal_backend PRIVATE "-framework Metal" "-framework Foundation")

install(TARGETS metal_backend LIBRARY DESTINATION .)
