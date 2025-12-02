from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch the simple navigation node
        Node(
            package='go2_simple_navigation',
            executable='goal_avoid',
            name='go2_simple_navigation',
            output='screen'
        )
    ])
