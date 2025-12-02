#!/usr/bin/env python3

import os
import launch_ros
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import Command, LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    description_path = LaunchConfiguration("description_path")
    world_path = LaunchConfiguration("world_path")
    spawn_x = LaunchConfiguration("spawn_x")
    spawn_y = LaunchConfiguration("spawn_y")
    spawn_z = LaunchConfiguration("spawn_z")
    spawn_yaw = LaunchConfiguration("spawn_yaw")
    enable_dwb = LaunchConfiguration("enable_dwb")
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", 
        default_value="true",  # Changed to true for Gazebo simulation
        description="Use simulation (Gazebo) clock if true"
    )
    
    # Robot description path
    pkg_share = launch_ros.substitutions.FindPackageShare(package="go2_description").find("go2_description")
    default_model_path = os.path.join(pkg_share, "xacro/robot.xacro")
    declare_description_path = DeclareLaunchArgument(
        name="description_path", 
        default_value=default_model_path, 
        description="Absolute path to robot urdf file"
    )
    
    # World file path - put your world file in go2_description/worlds/
    default_world_path = os.path.join(pkg_share, "worlds/obstacle_test_world.world")
    declare_world_path = DeclareLaunchArgument(
        name="world_path",
        default_value=default_world_path,
        description="Path to Gazebo world file"
    )
    
    # Robot spawn position arguments
    declare_spawn_x = DeclareLaunchArgument("spawn_x", default_value="-4.0", description="X spawn position")
    declare_spawn_y = DeclareLaunchArgument("spawn_y", default_value="-4.0", description="Y spawn position") 
    declare_spawn_z = DeclareLaunchArgument("spawn_z", default_value="0.5", description="Z spawn position")
    declare_spawn_yaw = DeclareLaunchArgument("spawn_yaw", default_value="0.0", description="Yaw spawn orientation")
    
    # Enable/disable DWB obstacle avoidance
    declare_enable_dwb = DeclareLaunchArgument("enable_dwb", default_value="true", description="Enable DWB obstacle avoidance")
    
    # Robot State Publisher (your existing configuration)
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {"robot_description": Command(["xacro ", description_path])},
            {"use_tf_static": False},
            {"publish_frequency": 200.0},
            {"ignore_timestamp": True},
            {'use_sim_time': use_sim_time}
        ],
    )
    
    # Joint State Publisher (needed for Gazebo simulation)
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # Launch Gazebo with the obstacle world
    gazebo_launch = ExecuteProcess(
        cmd=[
            'gazebo',
            '--verbose',
            world_path,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so'
        ],
        output='screen'
    )
    
    # Spawn Go2 robot in Gazebo (delayed to ensure Gazebo is ready)
    spawn_robot = TimerAction(
        period=3.0,  # Wait 3 seconds for Gazebo to start
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name='spawn_go2',
                output='screen',
                arguments=[
                    '-entity', 'go2_robot',
                    '-topic', '/robot_description',
                    '-x', spawn_x,
                    '-y', spawn_y,
                    '-z', spawn_z,
                    '-Y', spawn_yaw
                ]
            )
        ]
    )
    
    # DWB Obstacle Avoidance Node (conditional)
    dwb_obstacle_avoidance = Node(
        package='go2_description',  # Adjust if you put the script in a different package
        executable='dwb_obstacle_avoidance.py',  # Make sure this script is executable
        name='dwb_obstacle_avoidance',
        output='screen',
        condition=IfCondition(enable_dwb),
        parameters=[{
            # Go2-specific parameters
            'robot_radius': 0.35,  # Go2 robot radius (adjust based on actual size)
            'max_linear_vel': 1.2,  # Conservative for indoor testing
            'max_angular_vel': 1.8,
            'linear_acceleration': 0.8,
            'angular_acceleration': 1.5,
            
            # Detection parameters
            'detection_distance': 4.0,
            'min_height': -0.3,  # Adjust based on Go2 LiDAR height
            'max_height': 2.5,
            
            # Topic names (adjust based on your Go2 configuration)
            'lidar_topic': '/velodyne_points',  # Change if different
            'cmd_vel_topic': '/cmd_vel',
            'goal_topic': '/goal_point',
            
            # DWB algorithm parameters
            'prediction_time': 2.5,
            'dt': 0.1,
            'linear_samples': 8,
            'angular_samples': 16,
            
            # Scoring weights (tune for desired behavior)
            'obstacle_weight': 6.0,   # High priority for safety
            'goal_weight': 3.0,       # Medium priority for goal seeking
            'smoothness_weight': 1.5, # Low priority for smooth motion
            
            # Use simulation time
            'use_sim_time': use_sim_time
        }]
    )
    
    # Optional: Teleop node for manual control (useful for testing)
    teleop_node = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        prefix='xterm -e',  # Opens in separate terminal
        remappings=[('/cmd_vel', '/manual_cmd_vel')],  # Use different topic to avoid conflicts
        condition=IfCondition('false')  # Set to 'true' to enable teleop
    )
    
    # Optional: Goal publisher for testing (publishes a goal point)
    goal_publisher = Node(
        package='go2_description',
        executable='goal_publisher.py',  # You can create this simple script
        name='goal_publisher',
        output='screen',
        condition=IfCondition('false')  # Set to 'true' to enable automatic goal publishing
    )
    
    # Optional: RViz for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_share, 'rviz/go2_obstacle_avoidance.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition('false')  # Set to 'true' to enable RViz
    )
    
    return LaunchDescription([
        # Launch arguments
        declare_use_sim_time,
        declare_description_path,
        declare_world_path,
        declare_spawn_x,
        declare_spawn_y,
        declare_spawn_z,
        declare_spawn_yaw,
        declare_enable_dwb,
        
        # Core nodes
        gazebo_launch,
        robot_state_publisher_node,
        joint_state_publisher_node,
        spawn_robot,
        
        # Optional nodes (enable/disable as needed)
        dwb_obstacle_avoidance,
        # teleop_node,
        # goal_publisher,
        # rviz_node,
    ])
