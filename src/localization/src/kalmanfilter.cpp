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

    // Initialize the synchronizer.
    // The constructor takes the policy instance (with queue size) and the subscribers.
    // NOTE: The slop is implicitly handled by the queue size and internal algorithm,
    // or you can add a duration as a second argument to the policy constructor
    // if your message_filters version supports it directly.
    // For now, let's go with the queue size only, as that's generally sufficient.
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
    // Initialize state vector [x, y, theta, vx, vy, omega]
    state_ = Eigen::VectorXd::Zero(STATE_SIZE);
    covariance_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);

    Q_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    Q_ *= 0.1;  // Set process noise covariance to 0.1 for all state variables

    R_ = Eigen::MatrixXd::Identity(MEASUREMENT_SIZE, MEASUREMENT_SIZE);
    R_ *= 0.1;  // Set measurement noise covariance to 0.1 for all measurements

    A_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);

    B_ = Eigen::MatrixXd::Zero(STATE_SIZE, CONTROL_SIZE);
    
    u_ = Eigen::VectorXd::Zero(CONTROL_SIZE);

    H_ = Eigen::MatrixXd::Zero(MEASUREMENT_SIZE, STATE_SIZE);
    H_(0, 3) = 1.0;  // Measure vx
    H_(1, 4) = 1.0;  // Measure vy
    H_(2, 5) = 1.0;  // Measure omega
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
    state_ << odom_msg->pose.pose.position.x,     // x from odometry pose
             odom_msg->twist.twist.linear.x,      // vx from odometry
             odom_msg->pose.pose.position.y,      // y from odometry pose
             odom_msg->twist.twist.linear.y,      // vy from odometry
             0,                                   // theta from odometry quaternion
             imu_msg->angular_velocity.z;         // omega from IMU

    // Reduce initial uncertainty for velocities since we have measurements
    covariance_ *= 0.1;  // Set initial uncertainty to 0.1 for all state variables

    initialized_state_ = true;
    last_time_ = this->now();
    
    RCLCPP_INFO(this->get_logger(), 
                "Initialized state at origin with velocities: vx=%.3f, vy=%.3f, omega=%.3f",
                state_(3), state_(4), state_(5));
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

    // Update state transition matrix with current dt
    A_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    A_(1, 1) = 0; 
    A_(3, 3) = 0; 
    A_(5, 5) = 0;
    
    // Matrix looks like this
    // 1 0 0 0 0 0
    // 0 0 0 0 0 0
    // 0 0 1 0 0 0
    // 0 0 0 0 0 0
    // 0 0 0 0 1 0
    // 0 0 0 0 0 0

    // Update control input matrix
    double theta = state_(2);
    B_ = Eigen::MatrixXd::Zero(STATE_SIZE, CONTROL_SIZE);
    B_(0, 0) = cos(theta) * dt;
    B_(1, 0) = cos(theta); 
    B_(2, 0) = sin(theta) * dt; 
    B_(3, 0) = sin(theta); 
    B_(4, 1) = dt;
    B_(5, 1) = 1;

    // Standard Kalman Filter Prediction
    // x_k = A * x_{k-1} + B * u_k
    state_ = A_ * state_ + B_ * u_;
    
    // P_k = A * P_{k-1} * A^T + Q
    covariance_ = A_ * covariance_ * A_.transpose() + Q_;

    normalizeYaw();
    
    RCLCPP_DEBUG(this->get_logger(), 
                 "Predicted state: x=%.3f, y=%.3f, theta=%.3f (dt=%.4f)",
                 state_(0), state_(1), state_(2), dt);
  }

  void update(
      const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
      const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg)
  {
    // Create measurement vector - ONLY velocities!
    Eigen::VectorXd measurement(MEASUREMENT_SIZE);
    
    // Fill measurement vector with velocity data only
    measurement << odom_msg->twist.twist.linear.x,   // vx from odometry
                  odom_msg->twist.twist.linear.y,    // vy from odometry
                  imu_msg->angular_velocity.z;       // omega from IMU

    // Calculate innovation
    Eigen::VectorXd innovation = measurement - (H_ * state_);

    // Calculate Kalman gain
    Eigen::MatrixXd S = H_ * covariance_ * H_.transpose() + R_;
    Eigen::MatrixXd K = covariance_ * H_.transpose() * S.inverse();

    // Update state
    state_ = state_ + K * innovation;

    // Update covariance (Joseph form for numerical stability)
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    covariance_ = (I - K * H_) * covariance_;

    normalizeYaw();
    
    RCLCPP_DEBUG(this->get_logger(), 
                 "Updated with velocities: vx_meas=%.3f, vy_meas=%.3f, omega_meas=%.3f",
                 measurement(0), measurement(1), measurement(2));
  }

  void normalizeYaw()
  {
    while (state_(2) > M_PI) state_(2) -= 2 * M_PI;
    while (state_(2) < -M_PI) state_(2) += 2 * M_PI;
  }

  void publishFilteredState()
  {
    auto filtered_odom = nav_msgs::msg::Odometry();
    filtered_odom.header.stamp = this->now();
    
    // Use different frame IDs to avoid conflicts with original odometry
    filtered_odom.header.frame_id = "odom";
    filtered_odom.child_frame_id = "base_footprint";

    // Set position (integrated from velocities)
    filtered_odom.pose.pose.position.x = state_(0);
    filtered_odom.pose.pose.position.y = state_(1);
    filtered_odom.pose.pose.position.z = 0.0;

    // Set orientation
    tf2::Quaternion q;
    q.setRPY(0, 0, state_(2));
    filtered_odom.pose.pose.orientation.x = q.x();
    filtered_odom.pose.pose.orientation.y = q.y();
    filtered_odom.pose.pose.orientation.z = q.z();
    filtered_odom.pose.pose.orientation.w = q.w();

    // Set velocities (filtered)
    filtered_odom.twist.twist.linear.x = state_(3);
    filtered_odom.twist.twist.linear.y = state_(4);
    filtered_odom.twist.twist.angular.z = state_(5);

    // Set covariance matrices
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        // Pose covariance
        if (i < 3 && j < 3) {
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