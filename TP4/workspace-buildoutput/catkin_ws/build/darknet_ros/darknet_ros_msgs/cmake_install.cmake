# Install script for directory: /root/ros/catkin_ws/src/darknet_ros/darknet_ros_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/root/ros/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/darknet_ros_msgs/msg" TYPE FILE FILES
    "/root/ros/catkin_ws/src/darknet_ros/darknet_ros_msgs/msg/BoundingBox.msg"
    "/root/ros/catkin_ws/src/darknet_ros/darknet_ros_msgs/msg/BoundingBoxes.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/darknet_ros_msgs/action" TYPE FILE FILES "/root/ros/catkin_ws/src/darknet_ros/darknet_ros_msgs/action/CheckForObjects.action")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/darknet_ros_msgs/msg" TYPE FILE FILES
    "/root/ros/catkin_ws/devel/share/darknet_ros_msgs/msg/CheckForObjectsAction.msg"
    "/root/ros/catkin_ws/devel/share/darknet_ros_msgs/msg/CheckForObjectsActionGoal.msg"
    "/root/ros/catkin_ws/devel/share/darknet_ros_msgs/msg/CheckForObjectsActionResult.msg"
    "/root/ros/catkin_ws/devel/share/darknet_ros_msgs/msg/CheckForObjectsActionFeedback.msg"
    "/root/ros/catkin_ws/devel/share/darknet_ros_msgs/msg/CheckForObjectsGoal.msg"
    "/root/ros/catkin_ws/devel/share/darknet_ros_msgs/msg/CheckForObjectsResult.msg"
    "/root/ros/catkin_ws/devel/share/darknet_ros_msgs/msg/CheckForObjectsFeedback.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/darknet_ros_msgs/cmake" TYPE FILE FILES "/root/ros/catkin_ws/build/darknet_ros/darknet_ros_msgs/catkin_generated/installspace/darknet_ros_msgs-msg-paths.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/root/ros/catkin_ws/devel/include/darknet_ros_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/root/ros/catkin_ws/devel/share/roseus/ros/darknet_ros_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/root/ros/catkin_ws/devel/share/common-lisp/ros/darknet_ros_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/root/ros/catkin_ws/devel/share/gennodejs/ros/darknet_ros_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  execute_process(COMMAND "/usr/bin/python" -m compileall "/root/ros/catkin_ws/devel/lib/python2.7/dist-packages/darknet_ros_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/root/ros/catkin_ws/devel/lib/python2.7/dist-packages/darknet_ros_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/root/ros/catkin_ws/build/darknet_ros/darknet_ros_msgs/catkin_generated/installspace/darknet_ros_msgs.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/darknet_ros_msgs/cmake" TYPE FILE FILES "/root/ros/catkin_ws/build/darknet_ros/darknet_ros_msgs/catkin_generated/installspace/darknet_ros_msgs-msg-extras.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/darknet_ros_msgs/cmake" TYPE FILE FILES
    "/root/ros/catkin_ws/build/darknet_ros/darknet_ros_msgs/catkin_generated/installspace/darknet_ros_msgsConfig.cmake"
    "/root/ros/catkin_ws/build/darknet_ros/darknet_ros_msgs/catkin_generated/installspace/darknet_ros_msgsConfig-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/darknet_ros_msgs" TYPE FILE FILES "/root/ros/catkin_ws/src/darknet_ros/darknet_ros_msgs/package.xml")
endif()

