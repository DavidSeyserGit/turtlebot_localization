#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <Eigen/Dense>

class EKFNode : public rclcpp::Node
{
public:
    EKFNode() : Node("ekf_node")
    {
        // Initialize subscribers
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom", 10,
            std::bind(&EKFNode::odomCallback, this, std::placeholders::_1));
        
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu/data", 10,
            std::bind(&EKFNode::imuCallback, this, std::placeholders::_1));

        // Initialize publisher
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "ekf_pose", 10);

        // Initialize state vector and covariance matrix
        state_ = Eigen::VectorXd::Zero(6);  // [x, y, theta, v, omega, bias]
        covariance_ = Eigen::MatrixXd::Identity(6, 6);
        
        // Set initial covariance values
        covariance_.diagonal() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;

        RCLCPP_INFO(this->get_logger(), "EKF Node initialized");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // Update state with odometry data
        double dt = (msg->header.stamp.sec - last_odom_time_) * 1e-9;
        if (last_odom_time_ == 0)
        {
            last_odom_time_ = msg->header.stamp.sec;
            return;
        }

        // Extract measurements
        double v = msg->twist.twist.linear.x;
        double omega = msg->twist.twist.angular.z;

        // Predict step
        predict(dt, v, omega);

        // Update step with odometry
        updateOdom(msg);

        // Publish pose
        publishPose();

        last_odom_time_ = msg->header.stamp.sec;
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        // Update with IMU data
        double dt = (msg->header.stamp.sec - last_imu_time_) * 1e-9;
        if (last_imu_time_ == 0)
        {
            last_imu_time_ = msg->header.stamp.sec;
            return;
        }

        // Extract angular velocity
        double omega = msg->angular_velocity.z;

        // Update with IMU measurement
        updateImu(omega);

        last_imu_time_ = msg->header.stamp.sec;
    }

    void predict(double dt, double v, double omega)
    {
        // State transition matrix
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
        F(0, 3) = dt * cos(state_(2));
        F(1, 3) = dt * sin(state_(2));
        F(2, 4) = dt;

        // Process noise
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
        Q.diagonal() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;

        // Predict state
        state_(0) += v * dt * cos(state_(2));
        state_(1) += v * dt * sin(state_(2));
        state_(2) += omega * dt;

        // Update covariance
        covariance_ = F * covariance_ * F.transpose() + Q;
    }

    void updateOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // Measurement matrix
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 6);
        H(0, 0) = 1;  // x
        H(1, 1) = 1;  // y
        H(2, 2) = 1;  // theta

        // Measurement noise
        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3, 3);
        R.diagonal() << 0.1, 0.1, 0.1;

        // Measurement
        Eigen::VectorXd z(3);
        z << msg->pose.pose.position.x,
             msg->pose.pose.position.y,
             msg->pose.pose.orientation.z;

        // Kalman gain
        Eigen::MatrixXd K = covariance_ * H.transpose() * 
            (H * covariance_ * H.transpose() + R).inverse();

        // Update state and covariance
        state_ += K * (z - H * state_);
        covariance_ = (Eigen::MatrixXd::Identity(6, 6) - K * H) * covariance_;
    }

    void updateImu(double omega)
    {
        // Measurement matrix for IMU
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 6);
        H(0, 4) = 1;  // angular velocity

        // Measurement noise
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1) * 0.1;

        // Measurement
        Eigen::VectorXd z(1);
        z << omega;

        // Kalman gain
        Eigen::MatrixXd K = covariance_ * H.transpose() * 
            (H * covariance_ * H.transpose() + R).inverse();

        // Update state and covariance
        state_ += K * (z - H * state_);
        covariance_ = (Eigen::MatrixXd::Identity(6, 6) - K * H) * covariance_;
    }

    void publishPose()
    {
        auto pose_msg = geometry_msgs::msg::PoseWithCovarianceStamped();
        pose_msg.header.stamp = this->now();
        pose_msg.header.frame_id = "map";

        // Set position
        pose_msg.pose.pose.position.x = state_(0);
        pose_msg.pose.pose.position.y = state_(1);
        pose_msg.pose.pose.position.z = 0.0;

        // Set orientation (convert from yaw to quaternion)
        double yaw = state_(2);
        pose_msg.pose.pose.orientation.x = 0.0;
        pose_msg.pose.pose.orientation.y = 0.0;
        pose_msg.pose.pose.orientation.z = sin(yaw/2.0);
        pose_msg.pose.pose.orientation.w = cos(yaw/2.0);

        // Set covariance
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                pose_msg.pose.covariance[i*6 + j] = covariance_(i, j);
            }
        }

        pose_pub_->publish(pose_msg);
    }

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    
    // Publisher
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;

    // State vector and covariance
    Eigen::VectorXd state_;
    Eigen::MatrixXd covariance_;

    // Timestamps
    double last_odom_time_ = 0;
    double last_imu_time_ = 0;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 