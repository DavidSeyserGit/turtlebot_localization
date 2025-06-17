from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration # Add this
import os

def generate_launch_description():
    # Get package directories
    pkg_dir = get_package_share_directory('localization')
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')

    # Declare the 'use_sim_time' launch argument
    # It's good practice to declare arguments at the top level
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true', # Set default to true for simulation
        description='Use simulation time if true'
    )
    # Get the launch configuration for 'use_sim_time'
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Set Turtlebot3 model
    set_turtlebot_model = SetEnvironmentVariable(
        name='TURTLEBOT3_MODEL',
        value='waffle'
    )

    # Launch Turtlebot3 simulation
    # Pass 'use_sim_time' to the included launch file!
    turtlebot3_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_dir, 'launch', 'turtlebot3_world.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Add KalmanFilter node
    kalman_filter_node = Node(
        package='localization',
        executable='kalman_filter',
        name='kalman_filter',
        output='screen',
    )


    # Add teleop node for easy control
    teleop_node = Node(
        package='turtlebot3_teleop',
        executable='teleop_keyboard',
        name='teleop_keyboard',
        prefix = 'xterm -e',
        output='screen'
    )

    # Add dynamic reconfigure GUI
    rqt_reconfigure_node = Node(
        package='rqt_reconfigure',
        executable='rqt_reconfigure',
        name='rqt_reconfigure',
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg, # Declare it first
        set_turtlebot_model,
        turtlebot3_world,
        kalman_filter_node,
        teleop_node,
        rqt_reconfigure_node,
    ])