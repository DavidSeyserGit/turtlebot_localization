#include <cmath> // For M_PI
#include <random> // For noise generation

// ROS 2 Headers
#include "rclcpp/rclcpp.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

// ROS 2 Message Types
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"

// Eigen Library for Matrix Operations
#include <Eigen/Dense>

class KalmanFilter : public rclcpp::Node {
public:

  KalmanFilter() : Node("kalman_filter") {

    // Initialize Kalman filter parameters
    initializeKalmanFilter();

    // Set up synchronized subscribers for IMU and Joint States
    imu_sub_.subscribe(this, "/imu");
    joint_sub_.subscribe(this, "/joint_states");

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Imu, sensor_msgs::msg::JointState> MyApproxSyncPolicy;

    synchronizer_ = std::make_shared<message_filters::Synchronizer<MyApproxSyncPolicy>>(
        MyApproxSyncPolicy(100), // The policy object itself with queue size 100
        imu_sub_,
        joint_sub_
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
    
    RCLCPP_INFO(this->get_logger(), "Kalman filter initialized with wheel encoders and IMU. Waiting for messages...");
  }

private:
  // State vector: [x, vx, y, vy, theta, omega]
  Eigen::VectorXd state_;        // State vector
  Eigen::MatrixXd covariance_;   // State covariance
  Eigen::MatrixXd R_;            // Measurement noise covariance
  Eigen::MatrixXd Q_;            // Process noise covariance
  Eigen::MatrixXd A_;            // State transition matrix
  Eigen::MatrixXd B_;            // Control input matrix
  Eigen::MatrixXd C_;            // Observation matrix
  Eigen::VectorXd u_;            // Control input vector

  const int STATE_SIZE = 6;      // Size of state vector
  const int MEASUREMENT_SIZE = 3; // Only measuring [vx, vy, omega] - no ground truth pose!
  const int CONTROL_SIZE = 2;    // Size of control input vector [v, omega]

  bool initialized_state_ = false;
  rclcpp::Time last_time_;
  std::mutex state_mutex_;

  // Robot parameters
  const double WHEELBASE = 0.287;  // TurtleBot3 wheelbase in meters
  const double WHEEL_RADIUS = 0.033;  // TurtleBot3 wheel radius in meters

  // Synchronized subscribers
  message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
  message_filters::Subscriber<sensor_msgs::msg::JointState> joint_sub_;

  std::shared_ptr<message_filters::Synchronizer<
      message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Imu, sensor_msgs::msg::JointState>>> synchronizer_;

  // Regular subscribers
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr filtered_state_pub_;



  void initializeKalmanFilter() {
    // Initialize state vector
    state_ = Eigen::VectorXd::Zero(STATE_SIZE);
    
    // Initialize covariance matrix
    covariance_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    
    // Initialize process noise covariance
    Q_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE) * 0.1;
    
    // Initialize measurement noise covariance
    R_ = Eigen::MatrixXd::Identity(MEASUREMENT_SIZE, MEASUREMENT_SIZE);  // 3x3 for [vx, vy, omega]
    
    // Initialize state transition matrix
    A_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    
    // Initialize control input matrix
    B_ = Eigen::MatrixXd::Zero(STATE_SIZE, CONTROL_SIZE);
    
    // Initialize observation matrix - only measure velocities, not positions
    C_ = Eigen::MatrixXd::Zero(MEASUREMENT_SIZE, STATE_SIZE);  // 3x6 for measuring [vx, vy, omega]
    C_(0, 1) = 1.0;  // Measure vx (state index 1)
    C_(1, 3) = 1.0;  // Measure vy (state index 3) 
    C_(2, 5) = 1.0;  // Measure omega (state index 5)
    
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

  // Helper function to compute robot velocities from wheel encoder data
  std::pair<double, double> computeVelocitiesFromWheels(const sensor_msgs::msg::JointState::ConstSharedPtr joint_msg) 
  {
    // Find wheel velocities in joint state message
    double v_left = 0.0, v_right = 0.0;
    
    for (size_t i = 0; i < joint_msg->name.size(); ++i) {
      if (joint_msg->name[i] == "wheel_left_joint") {
        v_left = joint_msg->velocity[i] * WHEEL_RADIUS;  // Convert rad/s to m/s
      } else if (joint_msg->name[i] == "wheel_right_joint") {
        v_right = joint_msg->velocity[i] * WHEEL_RADIUS;  // Convert rad/s to m/s
      }
    }
    
    // Differential drive kinematics
    double v_linear = (v_left + v_right) / 2.0;          // Forward velocity
    double v_angular = (v_right - v_left) / WHEELBASE;   // Angular velocity from wheels
    
    return std::make_pair(v_linear, v_angular);
  }

  void synchronizedCallback(
      const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
      const sensor_msgs::msg::JointState::ConstSharedPtr joint_msg)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!initialized_state_) {
      initializeState(imu_msg, joint_msg);
      return;
    }

    predict();
    update(imu_msg, joint_msg);
    publishFilteredState();
  }

  void initializeState(
    const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
    const sensor_msgs::msg::JointState::ConstSharedPtr joint_msg)
  {
    // Compute initial velocities from wheel encoders
    auto wheel_velocities = computeVelocitiesFromWheels(joint_msg);
    double v_linear = wheel_velocities.first;
    
    // Initialize state with zero position and current velocities
    state_ << 0.0,                              // 0: x position (start at origin)
             v_linear,                          // 1: vx velocity (from wheels)
             0.0,                               // 2: y position (start at origin)
             0.0,                               // 3: vy velocity (assume no lateral motion)
             0.0,                               // 4: theta (start aligned with x-axis)
             imu_msg->angular_velocity.z;       // 5: omega (from IMU)

    // Set higher initial uncertainty for positions since we don't measure them
    covariance_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    covariance_(0, 0) = 1.0;   // High uncertainty in x position
    covariance_(1, 1) = 0.1;   // Lower uncertainty in vx (we measure it)
    covariance_(2, 2) = 1.0;   // High uncertainty in y position  
    covariance_(3, 3) = 0.1;   // Lower uncertainty in vy (assume small)
    covariance_(4, 4) = 0.5;   // Medium uncertainty in theta
    covariance_(5, 5) = 0.1;   // Lower uncertainty in omega (we measure it)

    initialized_state_ = true;
    last_time_ = this->now();
    
    RCLCPP_INFO(this->get_logger(), 
                "Initialized state from wheel encoders: x=%.3f, y=%.3f, theta=%.3f, vx=%.3f, vy=%.3f, omega=%.3f",
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

    // Store state before prediction for debugging
    Eigen::VectorXd state_before = state_;
    
    // Predict state
    state_ = A_ * state_ + B_ * u_;

    // Predict covariance
    covariance_ = A_ * covariance_ * A_.transpose() + Q_;

    normalizeYaw();
  }

  void update(
    const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
    const sensor_msgs::msg::JointState::ConstSharedPtr joint_msg)
  {
      // Compute velocities from wheel encoder data
      auto wheel_velocities = computeVelocitiesFromWheels(joint_msg);
      double vx_wheels = wheel_velocities.first;   // Forward velocity from wheels
      double vy_wheels = 0.0;                      // Assume no lateral velocity for diff drive robot

      // Create measurement vector z_t - velocities from wheel encoders and IMU
      Eigen::VectorXd z_t(MEASUREMENT_SIZE);  // [vx, vy, omega]
      z_t << vx_wheels,                       // vx from wheel encoders
            vy_wheels,                       // vy (zero for differential drive)
            imu_msg->angular_velocity.z;

      // Store predicted state and covariance (mu_bar_t, Sigma_bar_t)
      Eigen::VectorXd mu_bar_t = state_;
      Eigen::MatrixXd Sigma_bar_t = covariance_;

      // Step 4: K_t = Sigma_bar_t * C_t^T * (C_t * Sigma_bar_t * C_t^T + Q_t)^-1
      Eigen::MatrixXd S = C_ * Sigma_bar_t * C_.transpose() + R_;  // Innovation covariance
      Eigen::MatrixXd K_t = Sigma_bar_t * C_.transpose() * S.inverse();

      // Calculate C_t * mu_bar_t for step 5
      Eigen::VectorXd C_mu_bar = C_ * mu_bar_t;

      // Calculate innovation z_t - C_t * mu_bar_t (no angle normalization needed for velocities)
      Eigen::VectorXd innovation = z_t - C_mu_bar;

      // Step 5: mu_t = mu_bar_t + K_t * (z_t - C_t * mu_bar_t)
      Eigen::VectorXd mu_t = mu_bar_t + K_t * innovation;

      // Step 6: Sigma_t = (I - K_t * C_t) * Sigma_bar_t
      Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
      Eigen::MatrixXd Sigma_t = (I - K_t * C_) * Sigma_bar_t;

      // Rate-limited debug output (every 20 updates, ~1Hz at 20Hz)
      static int debug_counter = 0;
      if (++debug_counter % 20 == 0) {
        RCLCPP_INFO(this->get_logger(), 
                    "=== KALMAN FILTER UPDATE #%d ===", debug_counter);
        RCLCPP_INFO(this->get_logger(), 
                    "Measurements:  vx=%6.3f, vy=%6.3f, omega=%6.3f", 
                    z_t(0), z_t(1), z_t(2));
        RCLCPP_INFO(this->get_logger(), 
                    "Innovation:    vx=%6.3f, vy=%6.3f, omega=%6.3f", 
                    innovation(0), innovation(1), innovation(2));
        RCLCPP_INFO(this->get_logger(), 
                    "Kalman Gains:  vx=%6.3f, vy=%6.3f, omega=%6.3f", 
                    K_t(1,0), K_t(3,1), K_t(5,2));
        RCLCPP_INFO(this->get_logger(), 
                    "State: x=%6.3f, vx=%6.3f, y=%6.3f, vy=%6.3f, θ=%6.3f, ω=%6.3f",
                    mu_t(0), mu_t(1), mu_t(2), mu_t(3), mu_t(4), mu_t(5));
        RCLCPP_INFO(this->get_logger(), 
                    "Uncertainties: σ_x=%5.3f, σ_vx=%5.3f, σ_y=%5.3f, σ_vy=%5.3f, σ_θ=%5.3f, σ_ω=%5.3f",
                    sqrt(Sigma_t(0,0)), sqrt(Sigma_t(1,1)), sqrt(Sigma_t(2,2)), 
                    sqrt(Sigma_t(3,3)), sqrt(Sigma_t(4,4)), sqrt(Sigma_t(5,5)));
        RCLCPP_INFO(this->get_logger(), 
                    "=====================================");
      }

      // Update state and covariance
      state_ = mu_t;
      covariance_ = Sigma_t;

      normalizeYaw();
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