#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

class CmdVelPublisher : public rclcpp::Node
{
public:
  CmdVelPublisher()
  : Node("cmd_vel_publisher")
  {
    // Create a publisher on /cmd_vel, queue size 10
    publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10);

    // Publish at 10 Hz
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&CmdVelPublisher::on_timer, this));
  }

private:
  void on_timer()
  {
    auto msg = geometry_msgs::msg::Twist();
    // Set some velocities; here we go forward at 0.5 m/s
    msg.linear.x = 0.5;
    msg.linear.y = 0.0;
    msg.linear.z = 0.0;
    // No rotation
    msg.angular.x = 0.0;
    msg.angular.y = 0.0;
    msg.angular.z = 0.0;

    RCLCPP_INFO(this->get_logger(),
                "Publishing cmd_vel: linear.x=%.2f",
                msg.linear.x);
    publisher_->publish(msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdVelPublisher>());
  rclcpp::shutdown();
  return 0;
}
