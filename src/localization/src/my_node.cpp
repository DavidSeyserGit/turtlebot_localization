#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

class MyNode : public rclcpp::Node
{
  public:
    MyNode()
    : Node("cmd_vel_publisher")
    {
      // Create a publisher on /cmd_vel, queue size 10
      publisher_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
        "/cmd_vel", 10);

      scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10,
        std::bind(&MyNode::scan_callback, this, std::placeholders::_1)
      );

      // Publish at 10 Hz
      timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&MyNode::on_timer, this));
    }

  private:
    void on_timer()
    {
      // send cmd_vel message here
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr /*msg*/)
    {
      RCLCPP_INFO(this->get_logger(), "Received scan data");
    }

    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MyNode>());
  rclcpp::shutdown();
  return 0;
}
