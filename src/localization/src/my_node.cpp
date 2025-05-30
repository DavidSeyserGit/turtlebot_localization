#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include <Eigen/Dense>

#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class KalmanFilter : public rclcpp::Node
{
public:
  KalmanFilter() : Node("kalman_filter")
  {
    // Initialize Kalman filter parameters
    initializeKalmanFilter();

    imu_sub_.subscribe(this, "/imu");
    odom_sub_.subscribe(this, "/odom");

    synchronizer_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Imu, nav_msgs::msg::Odometry>>(imu_sub_, odom_sub_, 10);

    synchronizer_->registerCallback(
      std::bind(&KalmanFilter::synchronized_callback, this,
                std::placeholders::_1, std::placeholders::_2));
  }

private:
  // Kalman filter parameters
  Eigen::MatrixXd state_; // State vector -> estimated position and velocity
  Eigen::MatrixXd covariance_; // Covariance matrix -> uncertainty in the state estimate
  Eigen::MatrixXd R;
  Eigen::MatrixXd Q;
  Eigen::MatrixXd A; // State transition matrix
  Eigen::MatrixXd B; // Control input matrix
  Eigen::MatrixXd H; // Observation matrix
  Eigen::MatrixXd u; // Control input vector

  bool initialized_state_; // Flag to track if state has been initialized

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr initial_odom_sub_;

  void initializeKalmanFilter()
  {
    // Initialize matrices
    A = Eigen::MatrixXd::Identity(6, 6); // State transition matrix
    B = Eigen::MatrixXd::Zero(6, 3); // Control input matrix
    u = Eigen::MatrixXd::Zero(3, 1); // Control input vector (e.g., linear_velocity, angular_velocity)

    state_ = Eigen::MatrixXd::Zero(6, 1); // State vector: [x, y, theta, vx, vy, vtheta]
    covariance_ = Eigen::MatrixXd::Identity(6, 6);
    Q = Eigen::MatrixXd::Identity(6, 6) * 0.1; // Process noise covariance
  }

  void predict()
  {
    // x = A * x + B * u
    state_ = A * state_ + B * u; // Update state with the state transition matrix
    covariance_ = A * covariance_ * A.transpose() + Q; // Update covariance
  }

  void update()
  {

  }

  void synchronized_callback(
    const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
    const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg)
  {
    if (!initialized_state_)
    {
      // Initialize state with the first odometry message
      tf2::Quaternion q(
          odom_msg->pose.pose.orientation.x,
          odom_msg->pose.pose.orientation.y,
          odom_msg->pose.pose.orientation.z,
          odom_msg->pose.pose.orientation.w);
      tf2::Matrix3x3 m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);

      state_ << odom_msg->pose.pose.position.x, // x
            odom_msg->pose.pose.position.y, // y
            yaw,                            // yaw
            odom_msg->twist.twist.linear.x,
            odom_msg->twist.twist.linear.y,
            odom_msg->twist.twist.angular.z;
      covariance_ = Eigen::MatrixXd::Identity(6, 6); // Initialize covariance
      initialized_state_ = true;
    }
     // Log the current state for debugging
    RCLCPP_INFO(this->get_logger(), "Current State: [%f, %f, %f, %f, %f, %f]", state_(0), state_(1), state_(2), state_(3), state_(4), state_(5));
    predict();

    update();

    // Log the current state for debugging
    RCLCPP_INFO(this->get_logger(), "Current State: [%f, %f, %f, %f, %f, %f]", state_(0), state_(1), state_(2), state_(3), state_(4), state_(5));
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