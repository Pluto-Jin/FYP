project(resnetncnn)

cmake_minimum_required(VERSION 3.10)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20220216-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(resnetncnn SHARED res18_kd_ncnn_jni.cpp)

target_link_libraries(resnetncnn ncnn)
