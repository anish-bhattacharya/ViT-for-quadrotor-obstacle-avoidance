cmake_minimum_required(VERSION 3.0.0)
project(dodgeros_gui)

add_compile_options(-std=c++17)
add_compile_options(-O3)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  rqt_gui
  rqt_gui_py
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS geometry_msgs rospy rqt_gui rqt_gui_py std_msgs
)

install(FILES plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY resource
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
) 

install(PROGRAMS scripts/rqt_dodgeros_gui
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
