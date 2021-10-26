cmake_minimum_required(VERSION 2.8.8)

include_directories (${OpenCV_INCLUDE_DIRS})

#essas 3 linhas tem que ir pro outro projeto que for usar essa lib
find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_library(caffednn caffednn.cpp caffednn.h)
target_link_libraries(caffednn ${Caffe_LIBRARIES})

#find_library(CAFFE2_LIB caffe2)
#find_library(GLOG_LIB glog)
#find_library(GFLAGS_LIB gflags)

#if(NOT CAFFE2_LIB)
#  message(FATAL_ERROR "Caffe2 lib not found")
#endif()

#set(CAFFE2_LIBRARIES)
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

#macro(link_whole_archive lib)
#  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
#    list(APPEND CAFFE2_LIBRARIES -Wl,-force_load,$<TARGET_FILE_NAME:${lib}> ${lib})
#  elseif(MSVC)
#    list(APPEND CAFFE2_LIBRARIES -WHOLEARCHIVE:$<TARGET_FILE_NAME:${lib}>)
#  else()
#    list(APPEND CAFFE2_LIBRARIES -Wl,--whole-archive ${lib} -Wl,--no-whole-archive)
#  endif()
#endmacro()

#list(APPEND CAFFE2_LIBRARIES ${GLOG_LIB} ${GFLAGS_LIB})
#list(APPEND CAFFE2_LIBRARIES ${CAFFE2_LIB})

#add_library(caffe2dnn caffe2dnn.cpp caffe2dnn.h)
#link_whole_archive(caffe2)
#target_link_libraries(caffe2dnn ${CAFFE2_LIBRARIES} ${OpenCV_LIBRARIES})