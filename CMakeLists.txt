cmake_minimum_required(VERSION 3.16)

project(vision_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ORBBEC_SDK_ROOT
    "${CMAKE_SOURCE_DIR}/OrbbecSDK_C_C++_v1.10.37_20260707_3f75820_linux_x64_release/OrbbecSDK_v1.10.37"
    CACHE PATH "Orbbec SDK root directory")

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

if(NOT EXISTS "${ORBBEC_SDK_ROOT}/SDK/include")
    message(FATAL_ERROR
        "Orbbec SDK headers were not found under ${ORBBEC_SDK_ROOT}/SDK/include")
endif()

if(NOT EXISTS "${ORBBEC_SDK_ROOT}/SDK/lib/libOrbbecSDK.so")
    message(FATAL_ERROR
        "Orbbec SDK library was not found under ${ORBBEC_SDK_ROOT}/SDK/lib")
endif()

add_executable(orbbec_rgbd_capture
    native/orbbec_rgbd_capture.cpp)

add_executable(orbbec_multistream_probe
    tools/orbbec_multistream_probe.cpp)

foreach(target IN ITEMS orbbec_rgbd_capture orbbec_multistream_probe)
    target_include_directories(${target} PRIVATE
        "${ORBBEC_SDK_ROOT}/SDK/include")
    target_link_directories(${target} PRIVATE
        "${ORBBEC_SDK_ROOT}/SDK/lib")
    target_link_libraries(${target} PRIVATE
        OrbbecSDK
        pthread
        dl)
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${ORBBEC_SDK_ROOT}/SDK/lib")
endforeach()

target_link_libraries(orbbec_rgbd_capture PRIVATE ${OpenCV_LIBS})
