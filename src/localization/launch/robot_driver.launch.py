from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    pattern_arg = DeclareLaunchArgument(
        'pattern',
        default_value='circle',
        description='Driving pattern: circle, square, triangle, hexagon, spiral, sine_wave, zigzag, forward_circle_forward, figure8, random, straight, star'
    )
    
    linear_speed_arg = DeclareLaunchArgument(
        'linear_speed',
        default_value='0.2',
        description='Linear speed in m/s'
    )
    
    angular_speed_arg = DeclareLaunchArgument(
        'angular_speed',
        default_value='0.5',
        description='Angular speed in rad/s'
    )
    
    duration_arg = DeclareLaunchArgument(
        'duration',
        default_value='30.0',
        description='Duration of the pattern in seconds'
    )
    
    # Create the robot driver node
    robot_driver_node = Node(
        package='localization',
        executable='robot_driver',
        name='robot_driver',
        parameters=[{
            'pattern': LaunchConfiguration('pattern'),
            'linear_speed': LaunchConfiguration('linear_speed'),
            'angular_speed': LaunchConfiguration('angular_speed'),
            'duration': LaunchConfiguration('duration')
        }],
        output='screen'
    )
    
    return LaunchDescription([
        pattern_arg,
        linear_speed_arg,
        angular_speed_arg,
        duration_arg,
        robot_driver_node
    ])
