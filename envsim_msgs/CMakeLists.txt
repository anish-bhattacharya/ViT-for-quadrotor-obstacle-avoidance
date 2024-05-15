cmake_minimum_required(VERSION 3.0.0)
project(envsim_msgs)

add_compile_options(-std=c++17)
add_compile_options(-O3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_install()
cs_export()
