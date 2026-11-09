cmake_minimum_required(VERSION 3.16)
project(energy_m3t_offline LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
get_filename_component(ROOT "${ROOT}" ABSOLUTE)
set(M3T_SOURCE "${ROOT}/third_party/3DObjectTracking/M3T" CACHE PATH "Pinned upstream M3T source")
set(DEPS "${ROOT}/third_party/m3t_deps/root/usr" CACHE PATH "Project local build headers and GLFW")
find_package(OpenCV 4.3 REQUIRED COMPONENTS core imgproc highgui imgcodecs calib3d features2d videoio)
find_package(OpenMP REQUIRED)
# Build the upstream region backend without texture_modality (xfeatures2d),
# camera SDKs or examples. Upstream algorithm sources are left unmodified.
set(SOURCES common body renderer_geometry renderer normal_renderer silhouette_renderer
    basic_depth_renderer model region_model depth_model camera loader_camera viewer
    image_viewer normal_viewer color_histograms modality region_modality depth_modality
    link constraint soft_constraint optimizer detector static_detector manual_detector
    refiner publisher subscriber tracker)
list(TRANSFORM SOURCES PREPEND "${M3T_SOURCE}/src/")
list(TRANSFORM SOURCES APPEND ".cpp")
add_library(m3t_region STATIC ${SOURCES})
target_include_directories(m3t_region SYSTEM PUBLIC
    "${M3T_SOURCE}/include" "${M3T_SOURCE}/third_party" "${DEPS}/include"
    "${DEPS}/include/eigen3" ${OpenCV_INCLUDE_DIRS})
find_library(GLEW_LIB NAMES GLEW libGLEW.so.2.2 REQUIRED)
find_library(GL_LIB NAMES GL libGL.so.1 REQUIRED)
find_library(GLFW_LIB NAMES glfw glfw3 PATHS "${DEPS}/lib/x86_64-linux-gnu" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(m3t_region PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX
    "${GLEW_LIB}" "${GL_LIB}" "${GLFW_LIB}" pthread stdc++fs)
add_executable(m3t_offline m3t_offline.cpp)
target_link_libraries(m3t_offline PRIVATE m3t_region)
set_target_properties(m3t_offline PROPERTIES BUILD_RPATH "${DEPS}/lib/x86_64-linux-gnu")
add_executable(m3t_stream m3t_stream.cpp)
target_link_libraries(m3t_stream PRIVATE m3t_region)
set_target_properties(m3t_stream PROPERTIES BUILD_RPATH "${DEPS}/lib/x86_64-linux-gnu")
