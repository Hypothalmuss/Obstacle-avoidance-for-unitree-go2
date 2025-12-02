#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
import math
import numpy as np
from tf_transformations import euler_from_quaternion

class ObstacleAvoidanceNavigation(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_navigation')
       
        self.declare_parameter('goal_x', 5.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tolerance', 0.0)
        self.declare_parameter('detection_distance', 3.0)
        self.declare_parameter('detection_angle', 130.0)
        self.declare_parameter('min_height', -0.3)
        self.declare_parameter('max_height', 2.0)
        self.declare_parameter('min_points_threshold', 5)
        self.declare_parameter('max_linear_speed', 2.2)
        self.declare_parameter('max_angular_speed', 1.5)
        
        self.declare_parameter('safety_distance', 1.2)
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.detection_distance = self.get_parameter('detection_distance').value
        self.detection_angle = math.radians(self.get_parameter('detection_angle').value / 2.0)
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.min_points_threshold = self.get_parameter('min_points_threshold').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value
        
        self.declare_parameter('lidar_topic', '/velodyne_points')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        
        lidar_topic = self.get_parameter('lidar_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        
        self.lidar_sub = self.create_subscription(PointCloud2, lidar_topic, self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.obstacles = []
        self.goal_reached = False
        

        self.start_x = 0.0
        self.start_y = 0.0
        
	#robots states for chasing goal feature
        self.avoiding_obstacle = False  
        self.recovery_mode = False      
        self.last_goal_angle = 0.0      
        
        self.get_logger().info(f"Navigation node started")
        self.get_logger().info(f"Goal: ({self.goal_x}, {self.goal_y}) meters from start")

    def odom_callback(self, msg):
        """Update current robot position from odometry"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw angle
        orientation = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        # Set starting position on first odometry message
        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.get_logger().info(f"Starting position set: ({self.start_x:.2f}, {self.start_y:.2f})")

    def lidar_callback(self, msg):
        """Process lidar data to detect obstacles"""
        self.obstacles = []
        obstacle_points = []
        
        try:
            for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                x, y, z = point
                
                if z < self.min_height or z > self.max_height:
                    continue
                
                if x <= 0.01:
                    continue
                
                distance = math.sqrt(x**2 + y**2)
                angle = math.atan2(y, x)
                
                if distance <= self.detection_distance and abs(angle) <= self.detection_angle:
                    obstacle_points.append((x, y, distance, angle))
            
            self.obstacles = obstacle_points
            
        except Exception as e:
            self.get_logger().error(f"Error processing lidar: {e}")

    def is_path_to_goal_clear(self):
        if self.start_x is None:
            return False
            
        # Calculate goal position and angle
        goal_global_x = self.start_x + self.goal_x
        goal_global_y = self.start_y + self.goal_y
        goal_angle = math.atan2(goal_global_y - self.current_y, goal_global_x - self.current_x)
        
        goal_cone_angle = math.radians(60)  # ±45 degrees cone toward goal
        
        blocking_obstacles = []
        for obs in self.obstacles:
            obs_x, obs_y, obs_dist, obs_angle = obs
            
            # Check if obstacle is in the direction of the goal
            angle_diff = abs(obs_angle - (goal_angle - self.current_yaw))
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            angle_diff = abs(angle_diff)
            
            # If obstacle is in goal direction and close enough to matter
            if angle_diff < goal_cone_angle and obs_dist < self.safety_distance * 1.5:
                blocking_obstacles.append(obs)
        
        # Path is clear if no blocking obstacles
        return len(blocking_obstacles) == 0

    def control_loop(self):
        if self.start_x is None:
            return
        
        # Check if goal is reached
        goal_global_x = self.start_x + self.goal_x
        goal_global_y = self.start_y + self.goal_y
        distance_to_goal = math.sqrt((goal_global_x - self.current_x)**2 + (goal_global_y - self.current_y)**2)
        
        
        #########
        # Improvement test
        if distance_to_goal<1.0 :
        	self.set_parameters([
        	parameter('goal_tolerance', 0.05)
        	])
        #########
        
        
        if distance_to_goal < self.goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info("Goal reached!")
                self.goal_reached = True
            self.publish_stop_command()
            return
        
        cmd = Twist()
        immediate_obstacles = [obs for obs in self.obstacles if obs[2] < self.safety_distance]
        
        #decision logic
        if immediate_obstacles:
            cmd = self.avoid_obstacles(immediate_obstacles)
            self.avoiding_obstacle = True
            self.recovery_mode = False
            
        elif self.avoiding_obstacle and not self.is_path_to_goal_clear():
            cmd = self.continue_avoidance()
            self.recovery_mode = True
            
        else:
            cmd = self.seek_goal(goal_global_x, goal_global_y, distance_to_goal)
            
            # Reset avoidance flags
            if self.avoiding_obstacle or self.recovery_mode:
                self.get_logger().info("Recovery complete - returning to goal seeking")
            self.avoiding_obstacle = False
            self.recovery_mode = False
        
        # Publish command
        self.cmd_pub.publish(cmd)

    def avoid_obstacles(self, immediate_obstacles):
        cmd = Twist()
        self.get_logger().info(f"Avoiding {len(immediate_obstacles)} close obstacles")
        
        # Find the direction with shortest obstacle - comparing obstacls' length on the left and on the right
        left_obstacles = [obs for obs in immediate_obstacles if obs[3] > 0]
        right_obstacles = [obs for obs in immediate_obstacles if obs[3] < 0]
        front_obstacles = [obs for obs in immediate_obstacles if abs(obs[3]) < math.radians(20)]
        
        if len(front_obstacles) > 0:
            if len(left_obstacles) < len(right_obstacles):
                cmd.angular.z = self.max_angular_speed 
                self.get_logger().info("Turning LEFT to avoid obstacle")
            else:
                cmd.angular.z = -self.max_angular_speed 
                self.get_logger().info("Turning RIGHT to avoid obstacle")
            cmd.linear.x = 0.0
        else:
            # Adjust course slightly
            avg_angle = sum(obs[3] for obs in immediate_obstacles) / len(immediate_obstacles)
            cmd.angular.z = -avg_angle * 2.0
            cmd.linear.x = self.max_linear_speed 
        
        return cmd

    def continue_avoidance(self):
        """Continue avoidance when path to goal is not clear"""
        cmd = Twist()

        goal_global_x = self.start_x + self.goal_x
        goal_global_y = self.start_y + self.goal_y
        goal_angle = math.atan2(goal_global_y - self.current_y, goal_global_x - self.current_x)
        angle_to_goal = goal_angle - self.current_yaw

        while angle_to_goal > math.pi:
            angle_to_goal -= 2 * math.pi
        while angle_to_goal < -math.pi:
            angle_to_goal += 2 * math.pi
        

        cmd.angular.z = angle_to_goal * 0.8
        cmd.linear.x = 0.2  
        
        self.get_logger().info("Recovery mode - searching for clear path to goal")
        return cmd

    def seek_goal(self, goal_global_x, goal_global_y, distance_to_goal):
        """goal seeking logic"""
        cmd = Twist()

        goal_dx = goal_global_x - self.current_x
        goal_dy = goal_global_y - self.current_y
        goal_angle = math.atan2(goal_dy, goal_dx)

        angle_diff = goal_angle - self.current_yaw

        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        self.last_goal_angle = goal_angle

        cmd.angular.z = angle_diff * 1.5
        cmd.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, cmd.angular.z))

        if abs(angle_diff) > math.radians(30):
            cmd.linear.x = self.max_linear_speed * 0.3
        elif abs(angle_diff) > math.radians(10):
            cmd.linear.x = self.max_linear_speed * 0.6
        else:
            cmd.linear.x = self.max_linear_speed
        
        self.get_logger().info(f"Seeking goal: dist={distance_to_goal:.2f}m, angle_diff={math.degrees(angle_diff):.1f}°")
        return cmd

    def publish_stop_command(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ObstacleAvoidanceNavigation()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
