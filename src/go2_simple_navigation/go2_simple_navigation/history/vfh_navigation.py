#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math
import tf_transformations

class SimpleVFHNavigator(Node):
    def __init__(self):
        super().__init__('simple_vfh_navigator')
        
        # Simple publishers and subscribers - no QoS
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.scan_sub = self.create_subscription(PointCloud2, '/velodyne_points', self.scan_callback, 1)
        
        # Robot state
        self.current_pose = None
        self.obstacles = []
        
        # Simple parameters
        self.GOAL_X = 5.0
        self.GOAL_Y = 3.0
        self.SAFETY_DISTANCE = 1.5
        self.MAX_SPEED = 0.5
        self.GOAL_TOLERANCE = 0.3
        
        self.get_logger().info("Simple VFH Navigator started")
    
    def odom_callback(self, msg):
        """Get robot position"""
        self.current_pose = msg.pose.pose
        if not hasattr(self, 'odom_received'):
            self.get_logger().info("✓ Odometry received")
            self.odom_received = True
    
    def scan_callback(self, msg):
        """Get obstacles from point cloud"""
        obstacles = []
        try:
            point_count = 0
            for point in pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True):
                point_count += 1
                x, y, z = point
                # Only consider points at robot height and close enough
                if -1.0 < z < 2.0 and 0.1 < math.sqrt(x*x + y*y) < 8.0:
                    obstacles.append((x, y))
            
            self.obstacles = obstacles
            
            # Debug: Log obstacle info regularly
            if not hasattr(self, 'scan_count'):
                self.scan_count = 0
            self.scan_count += 1
            
            if self.scan_count % 20 == 1:  # Every 20 scans
                self.get_logger().info(f"Scan debug: {point_count} total points, {len(obstacles)} obstacles detected")
                if len(obstacles) > 0:
                    # Show closest obstacle
                    closest_obs = min(obstacles, key=lambda obs: math.sqrt(obs[0]**2 + obs[1]**2))
                    dist = math.sqrt(closest_obs[0]**2 + closest_obs[1]**2)
                    angle = math.degrees(math.atan2(closest_obs[1], closest_obs[0]))
                    self.get_logger().info(f"Closest obstacle: {dist:.2f}m at {angle:.0f}°")
                
        except Exception as e:
            self.get_logger().error(f"Scan error: {e}")
    
    def is_path_clear(self, target_angle):
        """Check if path in given direction is clear"""
        # If no obstacles detected, path is always clear
        if not self.obstacles:
            return True
            
        if not self.current_pose:
            return True
        
        obstacles_in_path = 0
        closest_obstacle_in_path = float('inf')
        
        # Check obstacles in the direction we want to go
        for obs_x, obs_y in self.obstacles:
            # Transform obstacle to robot frame
            obs_angle = math.atan2(obs_y, obs_x)
            obs_dist = math.sqrt(obs_x*obs_x + obs_y*obs_y)
            
            # Check if obstacle is in our path
            angle_diff = abs(obs_angle - target_angle)
            if angle_diff > math.pi:
                angle_diff = 2*math.pi - angle_diff
            
            # If obstacle is close and in our direction, path is blocked
            if obs_dist < self.SAFETY_DISTANCE and angle_diff < math.pi/4:  # 45 degree cone
                obstacles_in_path += 1
                closest_obstacle_in_path = min(closest_obstacle_in_path, obs_dist)
        
        is_clear = obstacles_in_path == 0
        
        # Debug: Always log for empty world debugging
        if len(self.obstacles) == 0:
            if not hasattr(self, 'empty_world_logged'):
                self.get_logger().info("🌍 Empty world detected - all paths should be clear")
                self.empty_world_logged = True
        
        return is_clear
    
    def find_clear_direction(self, goal_angle):
        """Find a clear direction to move"""
        # Debug: Always log what we're checking
        self.get_logger().info(f"Checking goal direction: {math.degrees(goal_angle):.0f}°, "
                             f"Total obstacles: {len(self.obstacles)}")
        
        # First try the goal direction
        if self.is_path_clear(goal_angle):
            return goal_angle
        
        self.get_logger().info("🚧 Goal direction blocked! Searching for alternative...")
        
        # Try directions around the goal angle - WIDER offsets
        for offset in [0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 2.5, -2.5]:
            test_angle = goal_angle + offset
            if self.is_path_clear(test_angle):
                self.get_logger().info(f"Found clear direction: {math.degrees(test_angle):.0f}°")
                return test_angle
        
        # If all else fails, try turning around
        self.get_logger().warn("All paths blocked! Attempting to turn around...")
        return goal_angle + math.pi
    
    def navigate(self):
        """Main navigation function"""
        if not self.current_pose:
            self.get_logger().info("Waiting for odometry...")
            return
        
        # Get current position and orientation
        pos = self.current_pose.position
        orient = self.current_pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])
        
        # Calculate distance and angle to goal
        dx = self.GOAL_X - pos.x
        dy = self.GOAL_Y - pos.y
        distance_to_goal = math.sqrt(dx*dx + dy*dy)
        
        # Check if we reached the goal
        if distance_to_goal < self.GOAL_TOLERANCE:
            self.get_logger().info("🎯 Goal reached!")
            cmd = Twist()  # Stop
            self.cmd_pub.publish(cmd)
            return
        
        # Calculate goal angle relative to robot
        goal_angle_global = math.atan2(dy, dx)
        goal_angle = goal_angle_global - yaw
        
        # Normalize angle
        while goal_angle > math.pi:
            goal_angle -= 2*math.pi
        while goal_angle < -math.pi:
            goal_angle += 2*math.pi
        
        # Find clear direction
        clear_direction = self.find_clear_direction(goal_angle)
        
        # Create movement command
        cmd = Twist()
        
        # Angular velocity - turn towards clear direction
        angular_error = clear_direction
        while angular_error > math.pi:
            angular_error -= 2*math.pi
        while angular_error < -math.pi:
            angular_error += 2*math.pi
        
        cmd.angular.z = max(-1.0, min(1.0, 2.0 * angular_error))
        
        # Linear velocity - slow down when turning
        if abs(angular_error) < 0.3:  # If pointing roughly in right direction
            cmd.linear.x = self.MAX_SPEED
        else:
            cmd.linear.x = self.MAX_SPEED * 0.3  # Slow down when turning
        
        self.cmd_pub.publish(cmd)
        
        # Log progress
        self.get_logger().info(f"Distance: {distance_to_goal:.1f}m, Goal angle: {math.degrees(goal_angle):.0f}°")

def main():
    rclpy.init()
    navigator = SimpleVFHNavigator()
    
    navigator.get_logger().info(f"🚀 Navigating to goal: ({navigator.GOAL_X}, {navigator.GOAL_Y})")
    
    try:
        while rclpy.ok():
            rclpy.spin_once(navigator, timeout_sec=0.1)
            navigator.navigate()
    except KeyboardInterrupt:
        navigator.get_logger().info("Navigation stopped")
    finally:
        # Stop the robot
        cmd = Twist()
        navigator.cmd_pub.publish(cmd)
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
