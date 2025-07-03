#include <cmath> // For M_PI
#include <mutex> // For std::mutex
#include <random> // For std::random_device, std::mt19937, std::normal_distribution

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
#include "geometry_msgs/msg/twist_stamped.hpp"

// Eigen Library for Matrix Operations
#include <Eigen/Dense>

class EKF_IMU : public rclcpp::Node {
public:

  EKF_IMU() : Node("ekf_imu") {

    // Load noise parameters with good defaults for IMU+Joint EKF
    process_noise_x_ = this->declare_parameter("process_noise_x", 0.1);
    process_noise_vx_ = this->declare_parameter("process_noise_vx", 0.1);
    process_noise_y_ = this->declare_parameter("process_noise_y", 0.1);
    process_noise_vy_ = this->declare_parameter("process_noise_vy", 0.1);
    process_noise_theta_ = this->declare_parameter("process_noise_theta", 0.1);
    process_noise_omega_ = this->declare_parameter("process_noise_omega", 0.1);
    
    measurement_noise_vx_ = this->declare_parameter("measurement_noise_vx", 0.1);
    measurement_noise_vy_ = this->declare_parameter("measurement_noise_vy", 0.1);
    measurement_noise_omega_ = this->declare_parameter("measurement_noise_omega", 0.001);
    
    initial_covariance_pos_ = this->declare_parameter("initial_covariance_pos", 0.001);
    initial_covariance_vel_ = this->declare_parameter("initial_covariance_vel", 0.001);
    initial_covariance_theta_ = this->declare_parameter("initial_covariance_theta", 0.001);
    initial_covariance_omega_ = this->declare_parameter("initial_covariance_omega", 0.001);

    // Robot parameters for TurtleBot3
    wheelbase_ = this->declare_parameter("wheelbase", 0.287);
    wheel_radius_ = this->declare_parameter("wheel_radius", 0.033);

    // Log loaded parameters
    RCLCPP_INFO(this->get_logger(), "EKF-IMU+Joint initialized with parameters:");
    RCLCPP_INFO(this->get_logger(), "  Process noise: pos=%.4f, vel=%.4f, theta=%.4f, omega=%.4f",
                process_noise_x_, process_noise_vx_, process_noise_theta_, process_noise_omega_);
    RCLCPP_INFO(this->get_logger(), "  Measurement noise: vx=%.4f, vy=%.4f, omega=%.4f",
                measurement_noise_vx_, measurement_noise_vy_, measurement_noise_omega_);
    RCLCPP_INFO(this->get_logger(), "  Robot params: wheelbase=%.3f, wheel_radius=%.3f",
                wheelbase_, wheel_radius_);

    // Initialize EKF parameters
    initializeEKF();

    // Set up synchronized subscribers for IMU and Joint States
    imu_sub_.subscribe(this, "/imu");
    joint_sub_.subscribe(this, "/joint_states");

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Imu, sensor_msgs::msg::JointState> MyApproxSyncPolicy;

    synchronizer_ = std::make_shared<message_filters::Synchronizer<MyApproxSyncPolicy>>(
        MyApproxSyncPolicy(100),
        imu_sub_,
        joint_sub_
    );

    synchronizer_->registerCallback(std::bind(
        &EKF_IMU::synchronizedCallback, this, 
        std::placeholders::_1, std::placeholders::_2));

    // Subscribe to cmd_vel for control input
    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
      "/cmd_vel", 10, [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        u_(0) = msg->twist.linear.x;
        u_(1) = msg->twist.angular.z;
        RCLCPP_DEBUG(this->get_logger(), "Received cmd_vel: v=%.3f, omega=%.3f", u_(0), u_(1));
      });

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
    
    RCLCPP_INFO(this->get_logger(), "EKF-IMU+Joint ready. Waiting for synchronized messages...");
  }

private:
  // Extended state vector: [x, vx, y, vy, theta, omega]
  static constexpr int STATE_SIZE = 6;
  static constexpr int MEASUREMENT_SIZE = 3; // [vx_wheels, vy_wheels, omega_imu]
  static constexpr int CONTROL_SIZE = 2; // [v_cmd, omega_cmd]

  Eigen::VectorXd state_;        // State vector
  Eigen::MatrixXd covariance_;   // State covariance
  Eigen::MatrixXd Q_;            // Process noise covariance
  Eigen::MatrixXd R_;            // Measurement noise covariance
  Eigen::VectorXd u_;            // Control input vector

  // Noise parameters
  double process_noise_x_, process_noise_vx_, process_noise_y_, process_noise_vy_;
  double process_noise_theta_, process_noise_omega_;
  double measurement_noise_vx_, measurement_noise_vy_, measurement_noise_omega_;
  double initial_covariance_pos_, initial_covariance_vel_, initial_covariance_theta_;
  double initial_covariance_omega_;

  // Robot parameters
  double wheelbase_;
  double wheel_radius_;

  bool initialized_state_ = false;
  rclcpp::Time last_time_;
  rclcpp::Time current_sensor_time_;
  std::mutex state_mutex_;

  // Synchronized subscribers
  message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
  message_filters::Subscriber<sensor_msgs::msg::JointState> joint_sub_;

  std::shared_ptr<message_filters::Synchronizer<
      message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Imu, sensor_msgs::msg::JointState>>> synchronizer_;

  // Regular subscribers and publishers
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr filtered_state_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_subscriber_;

  void initializeEKF() {
    // Initialize state vector: [x, vx, y, vy, theta, omega]
    state_ = Eigen::VectorXd::Zero(STATE_SIZE);
    
    // Initialize covariance matrix
    covariance_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    covariance_(0, 0) = 0.1;   // x position
    covariance_(1, 1) = 0.01;  // vx (we measure it)
    covariance_(2, 2) = 0.1;   // y position  
    covariance_(3, 3) = 0.01;  // vy (assume small)
    covariance_(4, 4) = 0.1;   // theta
    covariance_(5, 5) = 0.01;  // omega (we measure it)
    
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
    R_(0, 0) = measurement_noise_vx_;     // vx from wheels
    R_(1, 1) = measurement_noise_vy_;     // vy from wheels
    R_(2, 2) = measurement_noise_omega_;  // omega from IMU
    
    RCLCPP_DEBUG(this->get_logger(), "Updated EKF noise parameters");
  }

  // Helper function to compute robot velocities from wheel encoder data
  std::pair<double, double> computeVelocitiesFromWheels(const sensor_msgs::msg::JointState::ConstSharedPtr joint_msg) {
    double v_left = 0.0, v_right = 0.0;
    
    for (size_t i = 0; i < joint_msg->name.size(); ++i) {
        if (joint_msg->name[i] == "wheel_left_joint") {
            v_left = joint_msg->velocity[i] * wheel_radius_;
        } else if (joint_msg->name[i] == "wheel_right_joint") {
            v_right = joint_msg->velocity[i] * wheel_radius_;
        }
    }
    
    // Differential drive kinematics - robot frame velocities
    double v_linear = (v_left + v_right) / 2.0;
    double v_angular = (v_right - v_left) / wheelbase_;
    
    return std::make_pair(v_linear, v_angular);
  } 

  void synchronizedCallback(
      const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
      const sensor_msgs::msg::JointState::ConstSharedPtr joint_msg) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Store the sensor timestamp for proper header alignment
    current_sensor_time_ = imu_msg->header.stamp;
    
    if (!initialized_state_) {
      initializeState(imu_msg, joint_msg);
      return;
    }

    predict(imu_msg);
    update(imu_msg, joint_msg);
    publishFilteredState();
  }

  void initializeState(
      const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
      const sensor_msgs::msg::JointState::ConstSharedPtr joint_msg) {
    
    // Compute initial velocities from wheel encoders (robot frame)
    auto wheel_velocities = computeVelocitiesFromWheels(joint_msg);
    double v_robot = wheel_velocities.first;
    double omega_wheels = wheel_velocities.second;
    
    // Initialize state with zero position and current velocities in global frame
    state_ << 0.0,                              // 0: x position (start at origin)
             v_robot,                           // 1: vx velocity (robot frame v converted to global)
             0.0,                               // 2: y position (start at origin)
             0.0,                               // 3: vy velocity (zero for forward motion)
             0.0,                               // 4: theta (start aligned with x-axis)
             imu_msg->angular_velocity.z;       // 5: omega (from IMU)

    // Set initial uncertainty
    covariance_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    covariance_(0, 0) = 0.01;   // x position
    covariance_(1, 1) = 0.01;   // vx (we measure it)
    covariance_(2, 2) = 0.01;   // y position  
    covariance_(3, 3) = 0.01;   // vy (assume small)
    covariance_(4, 4) = 0.01;   // theta
    covariance_(5, 5) = 0.01;   // omega (we measure it)
    
    initialized_state_ = true;
    last_time_ = current_sensor_time_;
    
    RCLCPP_INFO(this->get_logger(), "EKF initialized. Initial state: x=%.3f, y=%.3f, θ=%.3f, vx=%.3f, vy=%.3f, ω=%.3f",
                state_(0), state_(2), state_(4), state_(1), state_(3), state_(5));
  }

  void predict(const sensor_msgs::msg::Imu::ConstSharedPtr /*imu_msg*/) {
    double dt = (current_sensor_time_ - last_time_).seconds();
    last_time_ = current_sensor_time_;
    if (dt <= 0.0 || dt > 1.0) {
        RCLCPP_WARN(this->get_logger(), "Invalid dt=%.3f, skipping predict", dt);
        return;
    }

    // Current state
    double x     = state_(0);
    double vx    = state_(1);
    double y     = state_(2);
    double vy    = state_(3);
    double theta = state_(4);
    double omega = state_(5);

    // Non-linear motion model using control inputs
    // For differential drive robot, we use the commanded velocities
    double v_cmd = u_(0);  // Linear velocity command
    double omega_cmd = u_(1);  // Angular velocity command
    
    // Non-linear state transition
    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(STATE_SIZE);
    
    if (std::abs(omega_cmd) > 1e-6) {
        // Non-zero angular velocity - curved motion
        x_pred(0) = x + (v_cmd / omega_cmd) * (std::sin(theta + omega_cmd * dt) - std::sin(theta));
        x_pred(1) = v_cmd * std::cos(theta + omega_cmd * dt);
        x_pred(2) = y + (v_cmd / omega_cmd) * (-std::cos(theta + omega_cmd * dt) + std::cos(theta));
        x_pred(3) = v_cmd * std::sin(theta + omega_cmd * dt);
        x_pred(4) = theta + omega_cmd * dt;
        x_pred(5) = omega_cmd;
    } else {
        // Zero angular velocity - straight line motion
        x_pred(0) = x + v_cmd * std::cos(theta) * dt;
        x_pred(1) = v_cmd * std::cos(theta);
        x_pred(2) = y + v_cmd * std::sin(theta) * dt;
        x_pred(3) = v_cmd * std::sin(theta);
        x_pred(4) = theta;
        x_pred(5) = 0.0;
    }

    // Compute Jacobian of the non-linear motion model
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    
    if (std::abs(omega_cmd) > 1e-6) {
        // Jacobian for curved motion
        G(0, 4) = (v_cmd / omega_cmd) * (std::cos(theta + omega_cmd * dt) - std::cos(theta));  // ∂x/∂θ
        G(1, 4) = -v_cmd * std::sin(theta + omega_cmd * dt);  // ∂vx/∂θ
        G(2, 4) = (v_cmd / omega_cmd) * (std::sin(theta + omega_cmd * dt) - std::sin(theta));  // ∂y/∂θ
        G(3, 4) = v_cmd * std::cos(theta + omega_cmd * dt);   // ∂vy/∂θ
    } else {
        // Jacobian for straight line motion
        G(0, 4) = -v_cmd * std::sin(theta) * dt;  // ∂x/∂θ
        G(1, 4) = -v_cmd * std::sin(theta);       // ∂vx/∂θ
        G(2, 4) = v_cmd * std::cos(theta) * dt;   // ∂y/∂θ
        G(3, 4) = v_cmd * std::cos(theta);        // ∂vy/∂θ
    }

    // Update state and covariance
    state_ = x_pred;
    covariance_ = G * covariance_ * G.transpose() + Q_;

    normalizeAngle();

    RCLCPP_DEBUG(this->get_logger(),
                "EKF PREDICT → x=%.3f y=%.3f θ=%.3f vx=%.3f vy=%.3f ω=%.3f",
                state_(0), state_(2), state_(4), state_(1), state_(3), state_(5));
  }

  void update(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
              const sensor_msgs::msg::JointState::ConstSharedPtr joint_msg) {
    
    // Compute velocities from wheel encoder data (robot frame)
    auto wheel_velocities = computeVelocitiesFromWheels(joint_msg);
    double v_robot = wheel_velocities.first;   // Linear velocity in robot frame
    double omega_wheels = wheel_velocities.second;  // Angular velocity from wheels
    double omega_imu = imu_msg->angular_velocity.z; // Angular velocity from IMU

    // Current state for measurement model
    double theta = state_(4);
    
    // Non-linear measurement model: convert robot frame velocities to global frame
    // Measurement vector z = [vx_global, vy_global, omega_imu]
    Eigen::VectorXd z(MEASUREMENT_SIZE);
    z << v_robot * std::cos(theta),  // vx in global frame
         v_robot * std::sin(theta),  // vy in global frame
         omega_imu;                  // omega from IMU

    // Predicted measurement h(x) = [vx_pred, vy_pred, omega_pred] from state
    Eigen::VectorXd h_pred(MEASUREMENT_SIZE);
    h_pred << state_(1),  // predicted vx
              state_(3),  // predicted vy
              state_(5);  // predicted omega

    // Non-linear measurement Jacobian H
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(MEASUREMENT_SIZE, STATE_SIZE);
    H(0, 1) = 1.0;  // ∂vx_meas/∂vx = 1
    H(0, 4) = -v_robot * std::sin(theta);  // ∂vx_meas/∂θ = -v_robot * sin(θ)
    H(1, 3) = 1.0;  // ∂vy_meas/∂vy = 1
    H(1, 4) = v_robot * std::cos(theta);   // ∂vy_meas/∂θ = v_robot * cos(θ)
    H(2, 5) = 1.0;  // ∂omega_meas/∂omega = 1

    // Innovation
    Eigen::VectorXd innovation = z - h_pred;

    // Kalman gain
    Eigen::MatrixXd S = H * covariance_ * H.transpose() + R_;
    Eigen::MatrixXd K = covariance_ * H.transpose() * S.inverse();

    // Update state and covariance
    state_ = state_ + K * innovation;
    
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    covariance_ = (I - K * H) * covariance_;

    normalizeAngle();
    
    RCLCPP_DEBUG(this->get_logger(), 
                 "Updated with measurements: v_robot=%.3f, omega_wheels=%.3f, omega_imu=%.3f",
                 v_robot, omega_wheels, omega_imu);
  }

  void normalizeAngle() {
    // Normalize theta to [-pi, pi]
    while (state_(4) > M_PI) state_(4) -= 2 * M_PI;
    while (state_(4) < -M_PI) state_(4) += 2 * M_PI;
  }

  void publishFilteredState() {
    auto filtered_odom = nav_msgs::msg::Odometry();
    filtered_odom.header.stamp = current_sensor_time_;
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