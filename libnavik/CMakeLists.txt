cmake_minimum_required(VERSION 3.14)

include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

add_library(navik_interfaces INTERFACE)
target_include_directories(navik_interfaces INTERFACE include)
target_link_libraries(navik_interfaces INTERFACE Eigen3::Eigen)
