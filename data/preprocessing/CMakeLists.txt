cmake_minimum_required(VERSION 3.1)
project(sdf_gen)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_CGAL              "Use CGAL"           ON)

# find_package(LIBIGL REQUIRED QUIET)

include(libigl)

# Enable the target igl::glfw
igl_include(glfw)

# Add your project files
file(GLOB SRCFILES sdf_gen.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})
# target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw igl::cgal)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw)
