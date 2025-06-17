#include <cmath> // For M_PI
#include <mutex> // For std::mutex

// ROS 2 Headers
#include "rclcpp/rclcpp.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

// ROS 2 Message Types
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"

// Eigen Library for Matrix Operations
#include <Eigen/Dense>

class EKF_IMU : public rclcpp::Node {
public:

  EKF_IMU() : Node("ekf_imu") {

    // Load noise parameters with good defaults for IMU-only EKF
    process_noise_x_ = this->declare_parameter("process_noise_x", 0.01);
    process_noise_vx_ = this->declare_parameter("process_noise_vx", 0.1);
    process_noise_y_ = this->declare_parameter("process_noise_y", 0.01);
    process_noise_vy_ = this->declare_parameter("process_noise_vy", 0.1);
    process_noise_theta_ = this->declare_parameter("process_noise_theta", 0.001);
    process_noise_omega_ = this->declare_parameter("process_noise_omega", 0.1);
    
    measurement_noise_ax_ = this->declare_parameter("measurement_noise_ax", 0.1);
    measurement_noise_ay_ = this->declare_parameter("measurement_noise_ay", 0.1);
    measurement_noise_gz_ = this->declare_parameter("measurement_noise_gz", 0.01);
    
    initial_covariance_pos_ = this->declare_parameter("initial_covariance_pos", 0.1);
    initial_covariance_vel_ = this->declare_parameter("initial_covariance_vel", 0.1);
    initial_covariance_theta_ = this->declare_parameter("initial_covariance_theta", 0.1);
    initial_covariance_omega_ = this->declare_parameter("initial_covariance_omega", 0.1);

    // Log loaded parameters
    RCLCPP_INFO(this->get_logger(), "EKF-IMU initialized with parameters:");
    RCLCPP_INFO(this->get_logger(), "  Process noise: pos=%.4f, vel=%.4f, theta=%.4f, omega=%.4f",
                process_noise_x_, process_noise_vx_, process_noise_theta_, process_noise_omega_);
    RCLCPP_INFO(this->get_logger(), "  Measurement noise: ax=%.4f, ay=%.4f, gz=%.4f",
                measurement_noise_ax_, measurement_noise_ay_, measurement_noise_gz_);

    // Initialize EKF parameters
    initializeEKF();

    // Subscribe to IMU only
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/imu", 10, std::bind(&EKF_IMU::imuCallback, this, std::placeholders::_1));

    // Subscribe to cmd_vel for control input
    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10, std::bind(&EKF_IMU::cmdVelCallback, this, std::placeholders::_1));

    // Create publisher for filtered state
    filtered_state_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
      "/ekf_state", 10);

    last_time_ = this->now();
    
    // Set up parameter callback for dynamic reconfigure
    auto param_callback = [this](const std::vector<rclcpp::Parameter> & params) {
      auto result = rcl_interfaces::msg::SetParametersResult();
      result.successful = true;
      this->updateNoiseParameters();
      return result;
    };
    
    param_subscriber_ = this->add_on_set_parameters_callback(param_callback);
    
    RCLCPP_INFO(this->get_logger(), "EKF-IMU ready. Waiting for IMU messages...");
  }

private:
  // Extended state vector: [x, vx, y, vy, theta, omega]
  static constexpr int STATE_SIZE = 6;
  static constexpr int MEASUREMENT_SIZE = 3; // [ax, ay, gz]
  static constexpr int CONTROL_SIZE = 2; // [v_cmd, omega_cmd]

  Eigen::VectorXd state_;        // State vector
  Eigen::MatrixXd covariance_;   // State covariance
  Eigen::MatrixXd Q_;            // Process noise covariance
  Eigen::MatrixXd R_;            // Measurement noise covariance
  Eigen::VectorXd u_;            // Control input vector

  // Noise parameters
  double process_noise_x_, process_noise_vx_, process_noise_y_, process_noise_vy_;
  double process_noise_theta_, process_noise_omega_;
  double measurement_noise_ax_, measurement_noise_ay_, measurement_noise_gz_;
  double initial_covariance_pos_, initial_covariance_vel_, initial_covariance_theta_;
  double initial_covariance_omega_;

  bool initialized_state_ = false;
  rclcpp::Time last_time_;
  std::mutex state_mutex_;

  // ROS subscribers and publishers
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr filtered_state_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_subscriber_;

  void initializeEKF() {
    // Initialize state vector: [x, vx, y, vy, theta, omega]
    state_ = Eigen::VectorXd::Zero(STATE_SIZE);
    
    // Initialize covariance matrix
    covariance_ = Eigen::MatrixXd::Zero(STATE_SIZE, STATE_SIZE);
    covariance_(0, 0) = initial_covariance_pos_;    // x
    covariance_(1, 1) = initial_covariance_vel_;    // vx
    covariance_(2, 2) = initial_covariance_pos_;    // y
    covariance_(3, 3) = initial_covariance_vel_;    // vy
    covariance_(4, 4) = initial_covariance_theta_;  // theta
    covariance_(5, 5) = initial_covariance_omega_;  // omega
    
    // Initialize process noise covariance
    updateNoiseParameters();
    
    // Initialize control input
    u_ = Eigen::VectorXd::Zero(CONTROL_SIZE);
  }

  void updateNoiseParameters() {
    // Update process noise matrix Q
    Q_ = Eigen::MatrixXd::Zero(STATE_SIZE, STATE_SIZE);
    Q_(0, 0) = process_noise_x_;       // x
    Q_(1, 1) = process_noise_vx_;      // vx
    Q_(2, 2) = process_noise_y_;       // y
    Q_(3, 3) = process_noise_vy_;      // vy
    Q_(4, 4) = process_noise_theta_;   // theta
    Q_(5, 5) = process_noise_omega_;   // omega
    
    // Update measurement noise matrix R
    R_ = Eigen::MatrixXd::Zero(MEASUREMENT_SIZE, MEASUREMENT_SIZE);
    R_(0, 0) = measurement_noise_ax_;  // ax
    R_(1, 1) = measurement_noise_ay_;  // ay
    R_(2, 2) = measurement_noise_gz_;  // gz
    
    RCLCPP_DEBUG(this->get_logger(), "Updated EKF noise parameters");
  }

  void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    u_(0) = msg->linear.x;   // v_cmd
    u_(1) = msg->angular.z;  // omega_cmd
    
    RCLCPP_DEBUG(this->get_logger(), "Received cmd_vel: v=%.3f, omega=%.3f", u_(0), u_(1));
  }

  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!initialized_state_) {
      initializeState(imu_msg);
      return;
    }

    predict();
    update(imu_msg);
    publishFilteredState();
  }

  void initializeState(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg) {
    // Initialize state with zero position/velocity, zero orientation
    state_.setZero();
    
    initialized_state_ = true;
    last_time_ = this->now();
    
    RCLCPP_INFO(this->get_logger(), "EKF initialized. Initial state: x=%.3f, y=%.3f, θ=%.3f, vx=%.3f, vy=%.3f, ω=%.3f",
                state_(0), state_(2), state_(4), state_(1), state_(3), state_(5));
  }

  void predict() {
    rclcpp::Time current_time = this->now();
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    if (dt <= 0 || dt > 1.0) {
      RCLCPP_WARN(this->get_logger(), "Invalid dt: %.4f, skipping prediction", dt);
      return;
    }

    // Get current state
    double x = state_(0);
    double vx = state_(1);
    double y = state_(2);
    double vy = state_(3);
    double theta = state_(4);
    double omega = state_(5);

    // Nonlinear state prediction (motion model)
    Eigen::VectorXd state_pred(STATE_SIZE);
    
    // Position integration with current velocity
    state_pred(0) = x + vx * dt;  // x += vx * dt
    state_pred(2) = y + vy * dt;  // y += vy * dt
    
    // Velocity update from control input
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    state_pred(1) = vx + (u_(0) * cos_theta) * dt;  // vx update from v_cmd
    state_pred(3) = vy + (u_(0) * sin_theta) * dt;  // vy update from v_cmd
    
    // Orientation integration
    state_pred(4) = theta + omega * dt;  // theta += omega * dt
    
    // Angular velocity update from control
    state_pred(5) = omega + (u_(1) - omega) * 0.1 * dt;  // Simple omega tracking

    // Compute Jacobian F for linearization
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    F(0, 1) = dt;  // dx/dvx
    F(2, 3) = dt;  // dy/dvy
    F(4, 5) = dt;  // dtheta/domega
    
    // Velocity coupling with orientation
    F(1, 4) = -u_(0) * sin_theta * dt;  // dvx/dtheta
    F(3, 4) = u_(0) * cos_theta * dt;   // dvy/dtheta

    // Predict state and covariance
    state_ = state_pred;
    covariance_ = F * covariance_ * F.transpose() + Q_;

    normalizeAngle();
    
    RCLCPP_DEBUG(this->get_logger(), 
                 "Predicted: x=%.3f, y=%.3f, theta=%.3f, vx=%.3f, vy=%.3f, omega=%.3f (dt=%.4f)",
                 state_(0), state_(2), state_(4), state_(1), state_(3), state_(5), dt);
  }

  void update(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg) {
    // Measurement vector z = [ax, ay, gz]
    Eigen::VectorXd z(MEASUREMENT_SIZE);
    z << imu_msg->linear_acceleration.x,
         imu_msg->linear_acceleration.y,
         imu_msg->angular_velocity.z;

    // Predicted measurement h(x) = [integrated vx, integrated vy, omega]
    // Use accelerations to predict velocity changes
    rclcpp::Time current_time = this->now();
    static rclcpp::Time last_update_time = current_time;
    double dt = (current_time - last_update_time).seconds();
    last_update_time = current_time;
    
    if (dt > 0 && dt < 1.0) {
      // Integrate accelerations to velocity estimates
      Eigen::VectorXd h_pred(MEASUREMENT_SIZE);
      h_pred << state_(1) + z(0) * dt,  // vx + ax*dt
                state_(3) + z(1) * dt,  // vy + ay*dt  
                z(2);                   // omega directly

      // Measurement Jacobian H
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(MEASUREMENT_SIZE, STATE_SIZE);
      H(0, 1) = 1.0;  // observe vx
      H(1, 3) = 1.0;  // observe vy
      H(2, 5) = 1.0;  // observe omega

      // Innovation
      Eigen::VectorXd innovation = h_pred - Eigen::VectorXd::Zero(MEASUREMENT_SIZE);
      innovation << state_(1) + z(0) * dt - state_(1),  // velocity update from acceleration
                    state_(3) + z(1) * dt - state_(3),  // velocity update from acceleration
                    z(2) - state_(5);                   // omega innovation

      // Kalman gain
      Eigen::MatrixXd S = H * covariance_ * H.transpose() + R_;
      Eigen::MatrixXd K = covariance_ * H.transpose() * S.inverse();

      // Update state and covariance
      state_ = state_ + K * innovation;
      
      Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
      covariance_ = (I - K * H) * covariance_;

      normalizeAngle();
      
      RCLCPP_DEBUG(this->get_logger(), 
                   "Updated with IMU: ax=%.3f, ay=%.3f, gz=%.3f",
                   z(0), z(1), z(2));
    }
  }

  void normalizeAngle() {
    // Normalize theta to [-pi, pi]
    while (state_(4) > M_PI) state_(4) -= 2 * M_PI;
    while (state_(4) < -M_PI) state_(4) += 2 * M_PI;
  }

  void publishFilteredState() {
    auto filtered_odom = nav_msgs::msg::Odometry();
    filtered_odom.header.stamp = this->now();
    filtered_odom.header.frame_id = "odom";
    filtered_odom.child_frame_id = "base_footprint";

    // Set position
    filtered_odom.pose.pose.position.x = state_(0);  // x
    filtered_odom.pose.pose.position.y = state_(2);  // y
    filtered_odom.pose.pose.position.z = 0.0;

    // Set orientation
    tf2::Quaternion q;
    q.setRPY(0, 0, state_(4));  // theta
    filtered_odom.pose.pose.orientation.x = q.x();
    filtered_odom.pose.pose.orientation.y = q.y();
    filtered_odom.pose.pose.orientation.z = q.z();
    filtered_odom.pose.pose.orientation.w = q.w();

    // Set velocities
    filtered_odom.twist.twist.linear.x = state_(1);   // vx
    filtered_odom.twist.twist.linear.y = state_(3);   // vy
    filtered_odom.twist.twist.angular.z = state_(5);  // omega

    // Set covariance (simplified)
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        if (i < 3 && j < 3) {
          // Position covariance
          filtered_odom.pose.covariance[i * 6 + j] = (i == j) ? covariance_(i * 2, j * 2) : 0.0;
        }
        if (i >= 3 && j >= 3) {
          // Velocity covariance
          int state_i = (i - 3) * 2 + 1;
          int state_j = (j - 3) * 2 + 1;
          filtered_odom.twist.covariance[i * 6 + j] = (i == j) ? covariance_(state_i, state_j) : 0.0;
        }
      }
    }

    filtered_state_pub_->publish(filtered_odom);
    
    // Log periodically
    static int counter = 0;
    if (++counter % 50 == 0) {
      RCLCPP_INFO(this->get_logger(), 
                  "EKF State: x=%.3f, y=%.3f, θ=%.3f, vx=%.3f, vy=%.3f, ω=%.3f",
                  state_(0), state_(2), state_(4), state_(1), state_(3), state_(5));
    }
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EKF_IMU>());
  rclcpp::shutdown();
  return 0;
}