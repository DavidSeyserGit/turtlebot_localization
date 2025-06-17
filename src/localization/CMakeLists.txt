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

# Add KalmanFilter executable
add_executable(kalman_filter src/kalmanfilter.cpp)

ament_target_dependencies(kalman_filter
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  message_filters
  tf2
  tf2_geometry_msgs
)
target_link_libraries(kalman_filter Eigen3::Eigen)

# Add EKF node executable
add_executable(ekf_node src/ekf_node.cpp)

ament_target_dependencies(ekf_node
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  message_filters
  tf2
  tf2_geometry_msgs
)
target_link_libraries(ekf_node Eigen3::Eigen)

# Add robot driver executable
add_executable(robot_driver src/robot_driver.cpp)

ament_target_dependencies(robot_driver
  rclcpp
  geometry_msgs
)
target_link_libraries(robot_driver)

install(
  TARGETS 
    kalman_filter
    ekf_node
    robot_driver
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install RViz configuration
install(DIRECTORY
  rviz
  DESTINATION share/${PROJECT_NAME}/
)

# Install map files
install(DIRECTORY
  map
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
