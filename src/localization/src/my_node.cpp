#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include <Eigen/Dense>

#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"

class KalmanFilter : public rclcpp::Node
{
public:
  KalmanFilter() : Node("kalman_filter")
  {
    imu_sub_.subscribe(this, "/imu");
    odom_sub_.subscribe(this, "/odom");

    synchronizer_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Imu, nav_msgs::msg::Odometry>>(imu_sub_, odom_sub_, 10);

    synchronizer_->registerCallback(
      std::bind(&KalmanFilter::synchronized_callback, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "KalmanFilter node started. Synchronizing /imu and /odom.");
  }

private:
  void synchronized_callback(
    const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
    const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg)
  {
    RCLCPP_INFO(this->get_logger(),
                "--- Synchronized Data Received (Timestamp: %.4f) ---",
                rclcpp::Time(imu_msg->header.stamp).seconds());

    RCLCPP_INFO(this->get_logger(),
                "IMU Orientation: (x=%.2f, y=%.2f, z=%.2f, w=%.2f)",
                imu_msg->orientation.x, imu_msg->orientation.y,
                imu_msg->orientation.z, imu_msg->orientation.w);

    RCLCPP_INFO(this->get_logger(),
                "Odometry Pose: (x=%.2f, y=%.2f, z=%.2f)",
                odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y,
                odom_msg->pose.pose.position.z);
    RCLCPP_INFO(this->get_logger(), "----------------------------------------");
  }

  message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
  std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Imu, nav_msgs::msg::Odometry>>synchronizer_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KalmanFilter>());
  rclcpp::shutdown();
  return 0;
}