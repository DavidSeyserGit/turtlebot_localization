# Turtlebot Localization

This project implements localization for a Turtlebot using ROS2.

## Prerequisites

- ROS2 (Humble or later)
- Turtlebot3 packages

## Turtlebot3 Setup

```bash
# Install Turtlebot3 packages
sudo apt install ros-<ros-version>-turtlebot3 ros-<ros-version>-turtlebot3-simulations

# Set the Turtlebot3 model
export TURTLEBOT3_MODEL=waffle_pi

# Add to your ~/.bashrc for persistence
echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
```

## Getting Started

1. **Source your ROS2 environment:**
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Build the workspace:**
   ```bash
   colcon build
   source install/setup.bash
   ```

3. **Launch the system:**
   ```bash
   ros2 launch localization turtlebot_bringup.launch.py
   ```

4. **Run the robot driver node:**
   ```bash
   ros2 run localization robot_driver
   ```

The system will start with the Turtlebot3 and the localization components running. 