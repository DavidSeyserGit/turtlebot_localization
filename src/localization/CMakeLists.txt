cmake_minimum_required(VERSION 3.5)
project(localization)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(Eigen3 REQUIRED CONFIG)

add_executable(cmd_vel_publisher src/my_node.cpp)
ament_target_dependencies(
  cmd_vel_publisher
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  message_filters
  tf2
  tf2_geometry_msgs
)

install(
  TARGETS cmd_vel_publisher
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
