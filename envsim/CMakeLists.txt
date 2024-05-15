project(envsim)
cmake_minimum_required(VERSION 3.0.0)

find_package(catkin_simple REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

# Setup Default Build Type as Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

# Setup Architcture-specific Flags
if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
    message(STATUS "Using ARMv7 optimized flags!")
    set(CMAKE_CXX_ARCH_FLAGS " -Wno-psabi -march=armv7-a -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations")
elseif ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    message(STATUS "Using ARM aarch64 optimized flags!")
    set(CMAKE_CXX_ARCH_FLAGS " -Wno-psabi -march=armv8-a+crypto -mcpu=cortex-a57+crypto")
else ()
    if (NOT DEFINED CATKIN_DEVEL_PREFIX)
        set(CMAKE_CXX_ARCH_FLAGS " -march=native")
    endif ()
endif ()

add_definitions(-Wall -Wextra -Wpedantic)
# Setup General C++ Flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_STACK_ALLOCATION_LIMIT=1048576")
add_definitions(-DEIGEN_DONT_PARALLELIZE -DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

# Setup Release and Debug flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_ARCH_FLAGS} -O3 -Wall -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_ARCH_FLAGS} -fsanitize=address -O3 -Wall -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")

cs_add_executable(visionsim_node src/visionsim_node.cpp)

# Simulator stuff
target_link_libraries(visionsim_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  zmq
  zmqpp
  yaml-cpp
)

  

# Finish
cs_install()
cs_export()
