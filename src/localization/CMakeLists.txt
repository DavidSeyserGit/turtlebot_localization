cmake_minimum_required(VERSION 3.5)
project(localization)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(cmd_vel_publisher src/my_node.cpp)
ament_target_dependencies(
  cmd_vel_publisher
  rclcpp
  geometry_msgs
)

install(
  TARGETS cmd_vel_publisher
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
