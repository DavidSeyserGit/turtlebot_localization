cmake_minimum_required(VERSION 3.8)
project(localization)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(cmd_vel_publisher
  src/my_node.cpp
  src/FilterBase.cpp
)

ament_target_dependencies(
  cmd_vel_publisher
  rclcpp
  geometry_msgs
  sensor_msgs
)

target_include_directories(cmd_vel_publisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(cmd_vel_publisher Eigen3::Eigen)

install(
  TARGETS cmd_vel_publisher
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
