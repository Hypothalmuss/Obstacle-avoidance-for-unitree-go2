#!/usr/bin/env python3
import rclpy
from rclpy.parameter import Parameter
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import numpy as np
from tf_transformations import euler_from_quaternion

class ObstacleAvoidanceNavigation(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_navigation')
       
        # Navigation parameters
        self.declare_parameter('goal_x', 6.0)  # Goal position X
        self.declare_parameter('goal_y', 0.0)  # Goal position Y
        self.declare_parameter('goal_tolerance', 0.2)  # Distance tolerance to goal
       
        # Detection parameters - made more sensitive
        self.declare_parameter('detection_distance', 3.0)  
        self.declare_parameter('detection_angle', 140.0)    # Wider detection angle
        self.declare_parameter('min_points_threshold', 1)  # Minimum points to detect obstacle
       
        # Control parameters
        self.declare_parameter('max_linear_speed', 3.5)    # Slower for better control
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('obstacle_weight', 3.0)     # Higher obstacle avoidance weight
        self.declare_parameter('goal_weight', 1.0)        
        self.declare_parameter('safety_distance', 1.5)     # Distance to start avoiding
       
        # Get parameters
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.detection_distance = self.get_parameter('detection_distance').value
        self.detection_angle = math.radians(self.get_parameter('detection_angle').value / 2.0)  # Half angle
        self.min_points_threshold = self.get_parameter('min_points_threshold').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.obstacle_weight = self.get_parameter('obstacle_weight').value
        self.goal_weight = self.get_parameter('goal_weight').value
        self.safety_distance = self.get_parameter('safety_distance').value
       
        # Topic names
        self.declare_parameter('lidar_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
       
        lidar_topic = self.get_parameter('lidar_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        odom_topic = self.get_parameter('odom_topic').value
       
        # Subscribers and Publishers
        self.lidar_sub = self.create_subscription(
            LaserScan,
            lidar_topic,
            self.lidar_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
       
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
       
        # State variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.obstacles = []  # List of obstacles: [(x, y, distance, angle)]
        self.goal_reached = False
        self.obstacle_detected = False
        self.tolerance_updated = False  # Flag to update tolerance only once
       
        # Initialize goal relative to starting position
        self.start_x = None
        self.start_y = None
       
        self.get_logger().info(f"Navigation node started")
        self.get_logger().info(f"Goal: ({self.goal_x}, {self.goal_y}) meters from start")
        self.get_logger().info(f"Detection distance: {self.detection_distance}m")
        self.get_logger().info(f"Detection angle: ±{math.degrees(self.detection_angle):.1f}°")
        self.get_logger().info(f"Safety distance: {self.safety_distance}m")

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
        """Process 2D lidar data to detect obstacles"""
        self.obstacles = []
        obstacle_points = []
       
        try:
            # Process LaserScan data
            angle = msg.angle_min
            for i, range_val in enumerate(msg.ranges):
                # Skip invalid readings
                if math.isnan(range_val) or math.isinf(range_val):
                    angle += msg.angle_increment
                    continue
                
                # Skip readings outside min/max range
                if range_val < msg.range_min or range_val > msg.range_max:
                    angle += msg.angle_increment
                    continue
                
                # Convert polar to cartesian (robot frame)
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                
                # Skip points behind the robot
                if x <= 0.01:  # Small threshold to avoid robot's own body
                    angle += msg.angle_increment
                    continue
                
                distance = range_val
                
                # Check if point is within detection zone
                if distance <= self.detection_distance and abs(angle) <= self.detection_angle:
                    obstacle_points.append((x, y, distance, angle))
                
                angle += msg.angle_increment
           
            # Store all detected obstacle points
            self.obstacles = obstacle_points
            self.obstacle_detected = len(self.obstacles) >= self.min_points_threshold
           
            # Log obstacle detection
            if self.obstacle_detected:
                closest_dist = min(obs[2] for obs in self.obstacles)
                self.get_logger().info(f"Obstacles detected: {len(self.obstacles)} points, closest: {closest_dist:.2f}m")
           
        except Exception as e:
            self.get_logger().error(f"Error processing lidar: {e}")
            # For safety, assume obstacles if processing fails
            self.obstacle_detected = True

    def control_loop(self):
        """Main control loop with clearer obstacle avoidance logic"""
        if self.start_x is None:
            return  # Wait for odometry
       
        # Check if goal is reached
        goal_global_x = self.start_x + self.goal_x
        goal_global_y = self.start_y + self.goal_y
        distance_to_goal = math.sqrt(
            (goal_global_x - self.current_x)**2 +
            (goal_global_y - self.current_y)**2
        )
       
       
        #########
        # Improvement test
        if distance_to_goal < 1.0 and not self.tolerance_updated:
            self.set_parameters([
                Parameter('goal_tolerance', Parameter.Type.DOUBLE, 0.04)
            ])
            self.goal_tolerance = self.get_parameter('goal_tolerance').value
            self.tolerance_updated = True
            self.get_logger().info(f"Goal tolerance updated to: {self.goal_tolerance}")
        #########
       
       
        if distance_to_goal < self.goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info("Goal reached!")
                self.goal_reached = True
            self.publish_stop_command()
            return
       
        # Initialize command
        cmd = Twist()
       
        # Check for immediate obstacles (within safety distance)
        immediate_obstacles = [obs for obs in self.obstacles if obs[2] < self.safety_distance]
       
        if immediate_obstacles:
            # OBSTACLE AVOIDANCE BEHAVIOR
            self.get_logger().info(f"Avoiding {len(immediate_obstacles)} close obstacles")
           
            # Find the direction with fewest obstacles
            left_obstacles = [obs for obs in immediate_obstacles if obs[3] > 0]  # Positive angle = left
            right_obstacles = [obs for obs in immediate_obstacles if obs[3] < 0]  # Negative angle = right
            front_obstacles = [obs for obs in immediate_obstacles if abs(obs[3]) < math.radians(20)]
           
            # Decide turning direction
            if len(front_obstacles) > 0:
                # There are obstacles directly in front - must turn
                if len(left_obstacles) < len(right_obstacles):
                    # Turn left (positive angular velocity)
                    cmd.angular.z = self.max_angular_speed * 0.8
                    cmd.linear.x = self.max_linear_speed * 0.3  # Move slowly while turning
                    self.get_logger().info("Turning LEFT to avoid obstacle")
                else:
                    # Turn right (negative angular velocity)
                    cmd.angular.z = -self.max_angular_speed * 0.8
                    cmd.linear.x = self.max_linear_speed * 0.3  # Move slowly while turning
                    self.get_logger().info("Turning RIGHT to avoid obstacle")
            else:
                # No front obstacles, but side obstacles - adjust course slightly
                avg_angle = sum(obs[3] for obs in immediate_obstacles) / len(immediate_obstacles)
                cmd.angular.z = -avg_angle * 2.0  # Turn away from average obstacle direction
                cmd.linear.x = self.max_linear_speed * 0.8
                self.get_logger().info(f"Adjusting course, avg obstacle angle: {math.degrees(avg_angle):.1f}°")
       
        else:
            # GOAL SEEKING BEHAVIOR
            # Calculate angle to goal
            goal_dx = goal_global_x - self.current_x
            goal_dy = goal_global_y - self.current_y
            goal_angle = math.atan2(goal_dy, goal_dx)
           
            # Calculate angle difference (robot's orientation vs goal direction)
            angle_diff = goal_angle - self.current_yaw
           
            # Normalize angle to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
           
            # Set angular velocity to turn toward goal
            cmd.angular.z = angle_diff * 2  # Proportional control
            cmd.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, cmd.angular.z))
           
            # Set linear velocity (slower if need to turn a lot)
            if abs(angle_diff) > math.radians(30):
                cmd.linear.x = self.max_linear_speed * 0.3  # Slow down for sharp turns
            elif abs(angle_diff) > math.radians(10):
                cmd.linear.x = self.max_linear_speed * 0.6  # Moderate speed
            else:
                cmd.linear.x = self.max_linear_speed  # Full speed ahead
           
            self.get_logger().info(f"Seeking goal: dist={distance_to_goal:.2f}m, Tol={self.goal_tolerance: .2f}, angle_diff={math.degrees(angle_diff):.1f}°")
       
        # Publish command
        self.cmd_pub.publish(cmd)

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