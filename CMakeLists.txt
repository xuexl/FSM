cmake_minimum_required(VERSION 3.5)

project(FSM LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_PREFIX_PATH "C:/software/libtorch/debug-1.6/")

set(Torch_DIR "C:/software/libtorch/debug-1.6/share/cmake/Torch")
#set(TorchVision_DIR "C:/software/libtorch/torchvision/share/cmake/TorchVision")

find_package(Torch REQUIRED)
#find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)

#find_package(Qt5Core)

include_directories("C:/software/libtorch/debug-1.6/include")
#include_directories("C:/software/libtorch/torchvision/include/torchvision")
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories("C:/software/libtorch/debug-1.6/lib")
#link_directories("C:/software/libtorch/torchvision/lib")

add_executable(FSM
  main.cxx
  data.h
  data.cxx
  config.h
  cout.h
  net.h
  net.cxx
)

#target_link_libraries(FSM TorchVision::TorchVision)
target_link_libraries(FSM "${TORCH_LIBRARIES}"  ${OpenCV_LIBS})
set_property(TARGET FSM PROPERTY CXX_STANDARD 14)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET FSM
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:FSM>)
endif (MSVC)