#include <cmath> // For M_PI

// ROS 2 Headers
#include "rclcpp/rclcpp.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

// ROS 2 Message Types
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"

// Eigen Library for Matrix Operations
#include <Eigen/Dense>

class KalmanFilter : public rclcpp::Node {
public:

  KalmanFilter() : Node("kalman_filter") {

    // Initialize Kalman filter parameters
    initializeKalmanFilter();

    // Set up synchronized subscribers for IMU and Odometry
    imu_sub_.subscribe(this, "/imu");
    odom_sub_.subscribe(this, "/odom");

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Imu, nav_msgs::msg::Odometry> MyApproxSyncPolicy;

    synchronizer_ = std::make_shared<message_filters::Synchronizer<MyApproxSyncPolicy>>(
        MyApproxSyncPolicy(100), // The policy object itself with queue size 100
        imu_sub_,
        odom_sub_
    );

    synchronizer_->registerCallback(std::bind(
        &KalmanFilter::synchronizedCallback, this, 
        std::placeholders::_1, std::placeholders::_2));

    // Subscribe to cmd_vel separately (doesn't need synchronization)
    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10, std::bind(&KalmanFilter::cmdVelCallback, this, std::placeholders::_1));

    // Create publisher for filtered state (visualization only)
    filtered_state_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
      "/filtered_state", 10);

    last_time_ = this->now();
    
    RCLCPP_INFO(this->get_logger(), "Kalman filter initialized with velocity-only odometry. Waiting for messages...");
  }

private:
  // State vector: [x, vx, y, vy, theta, omega]
  Eigen::VectorXd state_;        // State vector
  Eigen::MatrixXd covariance_;   // State covariance
  Eigen::MatrixXd R_;            // Measurement noise covariance
  Eigen::MatrixXd Q_;            // Process noise covariance
  Eigen::MatrixXd A_;            // State transition matrix
  Eigen::MatrixXd B_;            // Control input matrix
  Eigen::MatrixXd H_;            // Observation matrix
  Eigen::VectorXd u_;            // Control input vector

  const int STATE_SIZE = 6;      // Size of state vector
  const int MEASUREMENT_SIZE = 3; // Only measuring [vx, vy, omega] now!
  const int CONTROL_SIZE = 2;    // Size of control input vector [v, omega]

  bool initialized_state_ = false;
  rclcpp::Time last_time_;
  std::mutex state_mutex_;

  // Synchronized subscribers
  message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;

  std::shared_ptr<message_filters::Synchronizer<
      message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Imu, nav_msgs::msg::Odometry>>> synchronizer_;

  // Regular subscribers
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr filtered_state_pub_;

  void initializeKalmanFilter() {
    // Initialize state vector
    state_ = Eigen::VectorXd::Zero(STATE_SIZE);
    
    // Initialize covariance matrix
    covariance_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE) * 0.1;
    
    // Initialize process noise covariance
    Q_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE) * 0.1;
    
    // Initialize measurement noise covariance
    R_ = Eigen::MatrixXd::Identity(6, 6) * 0.1;  // 6x6 for [x, y, theta, vx, vy, omega]
    
    // Initialize state transition matrix
    A_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    
    // Initialize control input matrix
    B_ = Eigen::MatrixXd::Zero(STATE_SIZE, CONTROL_SIZE);
    
    // Initialize observation matrix
    H_ = Eigen::MatrixXd::Identity(6, STATE_SIZE);  // 6x6 for measuring all states
    
    // Initialize control input vector
    u_ = Eigen::VectorXd::Zero(CONTROL_SIZE);
  }

  void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    u_(0) = msg->linear.x;   // v
    u_(1) = msg->angular.z;  // omega
    
    RCLCPP_DEBUG(this->get_logger(), "Received cmd_vel: v=%.3f, omega=%.3f", u_(0), u_(1));
  }

  void synchronizedCallback(
      const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
      const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!initialized_state_) {
      initializeState(imu_msg, odom_msg);
      return;
    }

    predict();
    update(imu_msg, odom_msg);
    publishFilteredState();
  }

  void initializeState(
    const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
    const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg)
  {
    // Extract yaw from quaternion
    tf2::Quaternion q(
      odom_msg->pose.pose.orientation.x,
      odom_msg->pose.pose.orientation.y,
      odom_msg->pose.pose.orientation.z,
      odom_msg->pose.pose.orientation.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    state_ << odom_msg->pose.pose.position.x,     // 0: x position
             odom_msg->twist.twist.linear.x,      // 1: vx velocity
             odom_msg->pose.pose.position.y,      // 2: y position
             odom_msg->twist.twist.linear.y,      // 3: vy velocity
             yaw,                                 // 4: theta (yaw)
             imu_msg->angular_velocity.z;         // 5: omega

    // Reduce initial uncertainty for velocities since we have measurements
    covariance_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE) * 0.1;

    initialized_state_ = true;
    last_time_ = this->now();
    
    RCLCPP_INFO(this->get_logger(), 
                "Initialized state: x=%.3f, y=%.3f, theta=%.3f, vx=%.3f, vy=%.3f, omega=%.3f",
                state_(0), state_(2), state_(4), state_(1), state_(3), state_(5));
  }

  void predict()
  {
    rclcpp::Time current_time = this->now();
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    if (dt <= 0 || dt > 1.0) {  // Sanity check for dt
      RCLCPP_WARN(this->get_logger(), "Invalid dt: %.4f, skipping prediction", dt);
      return;
    }

    // Get current state values
    double x = state_(0);
    double vx = state_(1);
    double y = state_(2);
    double vy = state_(3);
    double theta = state_(4);
    double omega = state_(5);

    // Update state transition matrix with current dt
    // Using linear approximation: cos(theta) ≈ 1, sin(theta) ≈ 0 for small angles
    A_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    A_(0, 1) = dt;  // x += vx * dt
    A_(2, 3) = dt;  // y += vy * dt
    A_(4, 5) = dt;  // theta += omega * dt
  

    // Update control input matrix
    // Using linear approximation for small angles
    B_ = Eigen::MatrixXd::Zero(STATE_SIZE, CONTROL_SIZE);
    B_(1, 0) = 1.0;  // vx = v (assuming small angles)
    B_(3, 0) = 0.0;  // vy = 0 (assuming small angles)
    B_(5, 1) = 1.0;  // omega = omega_cmd

    // Predict state
    state_ = A_ * state_ + B_ * u_;

    // Predict covariance
    covariance_ = A_ * covariance_ * A_.transpose() + Q_;

    normalizeYaw();
    
    RCLCPP_DEBUG(this->get_logger(), 
                 "Predicted state: x=%.3f, y=%.3f, theta=%.3f, vx=%.3f, vy=%.3f, omega=%.3f (dt=%.4f)",
                 state_(0), state_(2), state_(4), state_(1), state_(3), state_(5), dt);
  }

  void update(
      const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
      const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg)
  {
    // Extract yaw from quaternion for measurement
    tf2::Quaternion q(
      odom_msg->pose.pose.orientation.x,
      odom_msg->pose.pose.orientation.y,
      odom_msg->pose.pose.orientation.z,
      odom_msg->pose.pose.orientation.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    // Create measurement vector - position, velocities, and orientation
    Eigen::VectorXd measurement(6);  // [x, y, theta, vx, vy, omega]
    measurement << odom_msg->pose.pose.position.x,   // x position
                  odom_msg->pose.pose.position.y,   // y position
                  yaw,                              // theta
                  odom_msg->twist.twist.linear.x,   // vx from odometry
                  odom_msg->twist.twist.linear.y,   // vy from odometry
                  imu_msg->angular_velocity.z;      // omega from IMU

    // Update observation matrix to include all measurements
    H_ = Eigen::MatrixXd::Identity(6, STATE_SIZE);

    // Calculate innovation (measurement - prediction)
    Eigen::VectorXd innovation = measurement - H_ * state_;

    // Normalize angle difference (still needed for numerical stability)
    while (innovation(2) > M_PI) innovation(2) -= 2 * M_PI;
    while (innovation(2) < -M_PI) innovation(2) += 2 * M_PI;

    // Calculate Kalman gainhttps://github.com/DavidSeyserGit/turtlebot_localization
    // Update covariance (Joseph form for numerical stability)
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    covariance_ = (I - K * H_) * covariance_ * (I - K * H_).transpose() + K * R_ * K.transpose();

    normalizeYaw();
    
    RCLCPP_DEBUG(this->get_logger(), 
                 "Updated with measurements: vx=%.3f, vy=%.3f, omega=%.3f, yaw=%.3f",
                 measurement(3), measurement(4), measurement(5), yaw);
  }

  void normalizeYaw()
  {
    // Normalize theta (state index 4) to [-pi, pi]
    while (state_(4) > M_PI) state_(4) -= 2 * M_PI;
    while (state_(4) < -M_PI) state_(4) += 2 * M_PI;
  }

  // this is so that i can rosbag record the filtered state

  void publishFilteredState()
  {
    auto filtered_odom = nav_msgs::msg::Odometry();
    filtered_odom.header.stamp = this->now();
    
    // Use different frame IDs to avoid conflicts with original odometry
    filtered_odom.header.frame_id = "odom";
    filtered_odom.child_frame_id = "base_footprint";

    // Set position (integrated from velocities)
    filtered_odom.pose.pose.position.x = state_(0);  // x position
    filtered_odom.pose.pose.position.y = state_(2);  // y position
    filtered_odom.pose.pose.position.z = 0.0;

    // Set orientation
    tf2::Quaternion q;
    q.setRPY(0, 0, state_(4));
    filtered_odom.pose.pose.orientation.x = q.x();
    filtered_odom.pose.pose.orientation.y = q.y();
    filtered_odom.pose.pose.orientation.z = q.z();
    filtered_odom.pose.pose.orientation.w = q.w();

    // Set velocities (filtered)
    filtered_odom.twist.twist.linear.x = state_(1);  // vx
    filtered_odom.twist.twist.linear.y = state_(3);  // vy
    filtered_odom.twist.twist.angular.z = state_(5); // omega

    // Set covariance matrices
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        // Pose covariance
        if (i < 3 && j < 3) {
          filtered_odom.pose.covariance[i * 6 + j] = covariance_(2, 2);
        } else if (i == j) {
          filtered_odom.pose.covariance[i * 6 + j] = covariance_(i, j);
        } else if (i == 5 && j == 5) {  // yaw
          filtered_odom.pose.covariance[i * 6 + j] = covariance_(2, 2);
        } else if (i == j) {
          filtered_odom.pose.covariance[i * 6 + j] = 1e-9;
        }

        // Twist covariance
        if (i < 3 && j < 3) {
          filtered_odom.twist.covariance[i * 6 + j] = covariance_(i + 3, j + 3);
        } else if (i >= 3 && j >= 3) {
          if (i == 5 && j == 5) {  // omega_z
            filtered_odom.twist.covariance[i * 6 + j] = covariance_(5, 5);
          } else if (i == j) {
            filtered_odom.twist.covariance[i * 6 + j] = 1e-9;
          }
        }
      }
    }

    filtered_state_pub_->publish(filtered_odom);
    
    // Log the filtered state periodically for debugging
    static int counter = 0;
    if (++counter % 40 == 0) {  // Log every 2 seconds at ~20Hz
      RCLCPP_INFO(this->get_logger(), 
                  "Filtered state: x=%.3f, y=%.3f, theta=%.3f, vx=%.3f, vy=%.3f, omega=%.3f",
                  state_(0), state_(1), state_(2), state_(3), state_(4), state_(5));
    }
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KalmanFilter>());
  rclcpp::shutdown();
  return 0;
}