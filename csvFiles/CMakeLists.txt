cmake_minimum_required(VERSION 3.10)
project(pileup)

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/source/*.cpp)


include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(c.out ${SOURCES})
