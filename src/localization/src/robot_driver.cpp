#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include <cmath>
#include <random>
#include <chrono>
#include <functional>

class RobotDriver : public rclcpp::Node
{
public:
    RobotDriver() : Node("robot_driver"), phase_(0.0)
    {
        // Create publisher for cmd_vel
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel", 10);
        
        // Create timer to publish commands at 10Hz
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&RobotDriver::publishCommands, this));
        
        // Declare parameters for different driving patterns
        this->declare_parameter("pattern", "circle");
        this->declare_parameter("linear_speed", 0.2);
        this->declare_parameter("angular_speed", 0.5);
        this->declare_parameter("duration", 30.0);  // seconds
        
        start_time_ = this->now();
        
        RCLCPP_INFO(this->get_logger(), "Robot driver started! Available patterns:");
        RCLCPP_INFO(this->get_logger(), "  - circle: Drives in circles");
        RCLCPP_INFO(this->get_logger(), "  - square: Drives in a square pattern");
        RCLCPP_INFO(this->get_logger(), "  - triangle: Drives in triangular pattern");
        RCLCPP_INFO(this->get_logger(), "  - hexagon: Drives in hexagonal pattern");
        RCLCPP_INFO(this->get_logger(), "  - spiral: Spiral outward pattern");
        RCLCPP_INFO(this->get_logger(), "  - sine_wave: Sinusoidal path");
        RCLCPP_INFO(this->get_logger(), "  - zigzag: Zigzag pattern");
        RCLCPP_INFO(this->get_logger(), "  - forward_circle_forward: Forward, circle, forward (10s total)");
        RCLCPP_INFO(this->get_logger(), "  - figure8: Drives in figure-8 pattern");
        RCLCPP_INFO(this->get_logger(), "  - random: Random walk");
        RCLCPP_INFO(this->get_logger(), "  - straight: Forward and backward");
    }

private:
    void publishCommands()
    {
        auto twist_stamped = geometry_msgs::msg::TwistStamped();
        twist_stamped.header.stamp = this->now();
        twist_stamped.header.frame_id = "base_link";
        
        // Get parameters
        std::string pattern = this->get_parameter("pattern").as_string();
        double linear_speed = this->get_parameter("linear_speed").as_double();
        double angular_speed = this->get_parameter("angular_speed").as_double();
        double duration = this->get_parameter("duration").as_double();
        
        // Check if we should stop
        double elapsed = (this->now() - start_time_).seconds();
        if (elapsed > duration) {
            twist_stamped.twist.linear.x = 0.0;
            twist_stamped.twist.angular.z = 0.0;
            cmd_vel_pub_->publish(twist_stamped);
            RCLCPP_INFO(this->get_logger(), "Driving pattern complete! Stopping robot.");
            rclcpp::shutdown();
            return;
        }
        
        // Update phase for time-based patterns
        phase_ += 0.1;  // 10Hz * 0.1 = 1.0 per second
        
        if (pattern == "circle") {
            // Drive in circles
            twist_stamped.twist.linear.x = linear_speed;
            twist_stamped.twist.angular.z = angular_speed;
        }
        else if (pattern == "square") {
            // Drive in square pattern: forward 3s, turn 1s, repeat
            double cycle_time = fmod(elapsed, 4.0);
            if (cycle_time < 3.0) {
                twist_stamped.twist.linear.x = linear_speed;
                twist_stamped.twist.angular.z = 0.0;
            } else {
                twist_stamped.twist.linear.x = 0.0;
                twist_stamped.twist.angular.z = angular_speed * 2.0;  // Faster turns
            }
        }
        else if (pattern == "figure8") {
            // Figure-8 pattern using sinusoidal angular velocity
            twist_stamped.twist.linear.x = linear_speed;
            twist_stamped.twist.angular.z = angular_speed * sin(phase_ * 0.5);
        }
        else if (pattern == "random") {
            // Random walk
            static double last_change = 0.0;
            static geometry_msgs::msg::TwistStamped last_twist_stamped;
            
            if (elapsed - last_change > 2.0) {  // Change direction every 2 seconds
                static std::random_device rd;
                static std::mt19937 gen(rd());
                static std::uniform_real_distribution<> linear_dist(-linear_speed, linear_speed);
                static std::uniform_real_distribution<> angular_dist(-angular_speed, angular_speed);
                
                last_twist_stamped.twist.linear.x = linear_dist(gen);
                last_twist_stamped.twist.angular.z = angular_dist(gen);
                last_change = elapsed;
            }
            twist_stamped.twist = last_twist_stamped.twist;
        }
        else if (pattern == "straight") {
            // Forward for 5s, backward for 5s, repeat
            double cycle_time = fmod(elapsed, 10.0);
            if (cycle_time < 5.0) {
                twist_stamped.twist.linear.x = linear_speed;
            } else {
                twist_stamped.twist.linear.x = -linear_speed;
            }
            twist_stamped.twist.angular.z = 0.0;
        }
        else if (pattern == "zigzag") {
            // Zigzag pattern: sharp left/right turns
            double cycle_time = fmod(elapsed, 4.0);
            twist_stamped.twist.linear.x = linear_speed;
            if (cycle_time < 2.0) {
                twist_stamped.twist.angular.z = angular_speed * 1.5;  // Turn left
            } else {
                twist_stamped.twist.angular.z = -angular_speed * 1.5; // Turn right
            }
        }
        else if (pattern == "forward_circle_forward") {
            // Forward for 3.3s, circle for 3.3s, forward for 3.3s (10s total)
            if (elapsed < 3.33) {
                // Phase 1: Drive forward
                twist_stamped.twist.linear.x = linear_speed;
                twist_stamped.twist.angular.z = 0.0;
            } else if (elapsed < 6.66) {
                // Phase 2: Drive in circle
                twist_stamped.twist.linear.x = linear_speed;
                twist_stamped.twist.angular.z = angular_speed;
            } else {
                // Phase 3: Drive forward again
                twist_stamped.twist.linear.x = linear_speed;
                twist_stamped.twist.angular.z = 0.0;
            }
        }
        else {
            RCLCPP_WARN_ONCE(this->get_logger(), "Unknown pattern: %s, using circle", pattern.c_str());
            twist_stamped.twist.linear.x = linear_speed;
            twist_stamped.twist.angular.z = angular_speed;
        }
        
        // Publish command
        cmd_vel_pub_->publish(twist_stamped);
        
        // Log status every 2 seconds
        static double last_log = 0.0;
        if (elapsed - last_log > 2.0) {
            RCLCPP_INFO(this->get_logger(), 
                        "Pattern: %s, Time: %.1f/%.1f s, v=%.2f m/s, ω=%.2f rad/s",
                        pattern.c_str(), elapsed, duration, twist_stamped.twist.linear.x, twist_stamped.twist.angular.z);
            last_log = elapsed;
        }
    }
    
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time start_time_;
    double phase_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotDriver>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 