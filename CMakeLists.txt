cmake_minimum_required (VERSION 3.0)
project (CUDA-RNN)

# The version number.
set (CUDA-RNN_VERSION_MAJOR 2)
set (CUDA-RNN_VERSION_MINOR 0)

#    CMake Scripts dir
#set(CMAKE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/CMakeScripts)

#CMake module path for custom module finding
#set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

find_package (CUDA REQUIRED)
include_directories (${CUDA_INCLUDE_DIRS})
#include_directories (${CUDA_SDK_ROOT_DIR})
include_directories ("${CUDA_INCLUDE_DIRS}/../samples/common/inc")
#/usr/local/cuda/NVIDIA_CUDA-7.0_Samples/common/inc")
include_directories("${CUDA_INCLUDE_DIRS}/../usr/include/nvidia/gdk")
#find_library(/usr/local/cuda-7.5/usr/src/gdk/nvml/lib)
set(CUDA_SEPARABLE_COMPILATION ON)



# opencv
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OPENCV_INCLUDE_DIR} )
set(OpenCV_LIBRARIES ${OpenCV_LIBS})


include_directories(${PROJECT_SOURCE_DIR})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -lnvidia-ml")


#-lnvidia-ml -L $(NVML_LIB)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -rdynamic")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_20,code=sm_20
  -gencode arch=compute_20,code=sm_21
  -gencode arch=compute_30,code=sm_30
  -gencode arch=compute_35,code=sm_35
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_50,code=compute_50
  -rdc=true
  )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  )

#add_library(${EXAMPLE_NAME}.o OBJECT ${source})
#set(EXAMPLE_OBJ_LIB $<TARGET_OBJECTS:${EXAMPLE_NAME}.o>)

file(GLOB CUDA_RNN_CU_SOURCES  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
	#    "readData/*" 
	#"layers/*"
	#"dataAugmentation/*" 
	#"common/*" 
	"*"
	#"*.cu"
    #"*.cpp"
    # "*.cuh"
    # "*.h"
    #"net.*"
    )
    #  EXCLUDE_FROM_ALL "build/*")
MESSAGE(${CUDA_RNN_CU_SOURCES})
cuda_add_executable(CUDA-RNN ${CUDA_RNN_CU_SOURCES})
#cuda_add_library(CUDA-CNN STATIC ${CUDA_RNN_CPP_SOURCES})
target_link_libraries(CUDA-RNN
  ${OpenCV_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_LIBRARIES}	
  /usr/lib/libnvidia-ml.so
  # /usr/local/cuda/usr/src/gdk/nvml/lib/libnvidia-ml.so
  )
CUDA_ADD_CUBLAS_TO_TARGET(CUDA-RNN)

#add_subdirectory(net)
