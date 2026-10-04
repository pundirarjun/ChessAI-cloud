cmake_minimum_required(VERSION 3.16)
project(az_cpp_mcts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT DEFINED pybind11_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_result
  )
  if(_pybind11_result EQUAL 0 AND _pybind11_dir)
    set(pybind11_DIR "${_pybind11_dir}")
  endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(az_cpp_mcts src/python_module.cpp)
target_include_directories(az_cpp_mcts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
  target_compile_options(az_cpp_mcts PRIVATE /O2 /EHsc /W3 /fp:fast)
else()
  target_compile_options(az_cpp_mcts PRIVATE -O3 -Wall -Wextra)
endif()

# Emit the importable module at the repository root so `import az_cpp_mcts`
# works from the repo working directory (preflight checks this).
set(_out_root "${CMAKE_CURRENT_SOURCE_DIR}/..")
set_target_properties(az_cpp_mcts PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${_out_root}"
  RUNTIME_OUTPUT_DIRECTORY "${_out_root}"
)
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(az_cpp_mcts PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${_cfg} "${_out_root}"
    RUNTIME_OUTPUT_DIRECTORY_${_cfg} "${_out_root}"
  )
endforeach()
