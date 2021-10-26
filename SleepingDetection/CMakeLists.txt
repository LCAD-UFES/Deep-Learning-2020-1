cmake_minimum_required (VERSION 2.8)
project (sleeping_detection)
set(CAFFE_DIR ${PROJECT_SOURCE_DIR}/lib/caffe)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(Caffe REQUIRED)

include_directories (lib/caffe)
add_subdirectory (lib/caffe)
message(STATUS "CAFFE_DIR=" ${CAFFE_DIR})

add_executable (sleeping_detection src/sleeping_detection_main.cpp src/face_detector.cpp ${sleeping_detection_headers})
target_link_libraries (sleeping_detection caffednn ${Caffe_LIBRARIES} dlib::dlib ${OpenCV_LIBRARIES})