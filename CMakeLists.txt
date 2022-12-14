cmake_minimum_required(VERSION 2.8)

project( HelloWorld_Proj )

# init_qt: Let's do the CMake job for us
set(CMAKE_AUTOMOC ON) # For meta object compiler
set(CMAKE_AUTORCC ON) # Resource files
set(CMAKE_AUTOUIC ON) # UI files

find_package(Qt6 QUIET COMPONENTS Concurrent OpenGL Widgets)
if (NOT Qt6_FOUND)
  find_package(Qt5 COMPONENTS Concurrent OpenGL Widgets REQUIRED)
  set(QTX Qt5)
else()
  set(QTX Qt6)
endif()
find_package(VTK REQUIRED)

find_package ( OpenCV REQUIRED )
find_package(PCL 1.2 REQUIRED)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable ( Main_Filters src/main.cpp )
target_link_libraries ( Main_Filters ${OpenCV_LIBS} )

add_executable( Disparity_2_point_cloud src/disparity2pc.cpp )
target_link_libraries( Disparity_2_point_cloud ${OpenCV_LIBS} ${PCL_LIBRARIES})

set(project_SOURCES src/main_gui.cpp src/pclviewer.cpp)
add_executable(Main_GUI ${project_SOURCES})
target_link_libraries(Main_GUI ${OpenCV_LIBS} ${PCL_LIBRARIES} ${QTX}::Widgets)
