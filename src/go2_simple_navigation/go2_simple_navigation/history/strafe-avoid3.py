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

class StrafingObstacleAvoidanceNavigation(Node):
    def __init__(self):
        super().__init__('strafing_obstacle_avoidance_navigation')
       
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tolerance', 0.05)
        self.declare_parameter('detection_distance', 3.0)
        self.declare_parameter('detection_angle', 120.0)
        self.declare_parameter('min_height', -0.3)
        self.declare_parameter('max_height', 2.0)
        self.declare_parameter('min_points_threshold', 5)
        self.declare_parameter('max_linear_speed', 2.2)
        self.declare_parameter('max_strafe_speed', 1.5)
       
        # New parameters for the modified behavior
        self.declare_parameter('front_detection_angle', 60.0)  # Angle for front obstacle detection
        self.declare_parameter('side_detection_angle', 90.0)   # Angle for side obstacle detection
        self.declare_parameter('safety_distance', 1.2)
        self.declare_parameter('side_clearance_distance', 1.0)  # Distance to consider side clear
       
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.detection_distance = self.get_parameter('detection_distance').value
        self.detection_angle = math.radians(self.get_parameter('detection_angle').value / 2.0)
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.min_points_threshold = self.get_parameter('min_points_threshold').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_strafe_speed = self.get_parameter('max_strafe_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value
        
        # New parameters
        self.front_detection_angle = math.radians(self.get_parameter('front_detection_angle').value / 2.0)
        self.side_detection_angle = math.radians(self.get_parameter('side_detection_angle').value)
        self.side_clearance_distance = self.get_parameter('side_clearance_distance').value
       
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
       
        self.start_x = None  # Fixed: Should be None initially
        self.start_y = None
       
        # Robot states for the new behavior
        self.state = "SEEKING_GOAL"  # States: SEEKING_GOAL, STRAFING_AWAY, MOVING_BESIDE, STRAFING_BACK
        self.strafe_direction = 0  # -1 for right, 1 for left
        self.obstacle_cleared_position = None  # Position where we started moving beside obstacle
       
        self.get_logger().info(f"Strafing Navigation node started")
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

    def detect_front_obstacle(self):
        """Check if there's an obstacle directly in front"""
        front_obstacles = []
        for obs in self.obstacles:
            obs_x, obs_y, obs_dist, obs_angle = obs
            if abs(obs_angle) <= self.front_detection_angle and obs_dist <= self.safety_distance:
                front_obstacles.append(obs)
        return len(front_obstacles) > 0, front_obstacles

    def analyze_strafe_options(self):
        """Analyze which side is better for strafing"""
        left_obstacles = []
        right_obstacles = []
        
        for obs in self.obstacles:
            obs_x, obs_y, obs_dist, obs_angle = obs
            
            # Left side (positive angle)
            if obs_angle > self.front_detection_angle and obs_angle <= self.side_detection_angle:
                if obs_dist <= self.safety_distance * 1.5:  # Slightly larger detection for side planning
                    left_obstacles.append(obs)
            
            # Right side (negative angle)
            elif obs_angle < -self.front_detection_angle and obs_angle >= -self.side_detection_angle:
                if obs_dist <= self.safety_distance * 1.5:
                    right_obstacles.append(obs)
        
        # Choose side with fewer obstacles, prefer right if equal
        if len(left_obstacles) <= len(right_obstacles):
            return 1, left_obstacles  # Strafe left (positive y direction)
        else:
            return -1, right_obstacles  # Strafe right (negative y direction)

    def is_front_clear(self):
        """Check if front is clear enough to move forward"""
        has_front_obstacle, _ = self.detect_front_obstacle()
        return not has_front_obstacle

    def is_side_clear(self):
        """Check if the side we're avoiding is now clear"""
        if self.strafe_direction == 0:
            return True
            
        side_obstacles = []
        target_angle_range = self.side_detection_angle if self.strafe_direction == 1 else -self.side_detection_angle
        
        for obs in self.obstacles:
            obs_x, obs_y, obs_dist, obs_angle = obs
            
            if self.strafe_direction == 1:  # We strafed left, check right side for clearance
                if obs_angle >= -self.side_detection_angle and obs_angle <= -self.front_detection_angle:
                    if obs_dist <= self.side_clearance_distance:
                        side_obstacles.append(obs)
            else:  # We strafed right, check left side for clearance
                if obs_angle <= self.side_detection_angle and obs_angle >= self.front_detection_angle:
                    if obs_dist <= self.side_clearance_distance:
                        side_obstacles.append(obs)
        
        return len(side_obstacles) == 0

    def get_angle_to_goal(self):
        """Calculate angle difference to goal"""
        if self.start_x is None:
            return 0.0
            
        goal_global_x = self.start_x + self.goal_x
        goal_global_y = self.start_y + self.goal_y
        goal_angle = math.atan2(goal_global_y - self.current_y, goal_global_x - self.current_x)
        
        angle_diff = goal_angle - self.current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        return angle_diff

    def control_loop(self):
        if self.start_x is None:
            return
       
        # Check if goal is reached
        goal_global_x = self.start_x + self.goal_x
        goal_global_y = self.start_y + self.goal_y
        distance_to_goal = math.sqrt((goal_global_x - self.current_x)**2 + (goal_global_y - self.current_y)**2)
       
        if distance_to_goal <= self.goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info("Goal reached!")
                self.goal_reached = True
            self.publish_stop_command()
            return

        cmd = Twist()
        
        # State machine for obstacle avoidance behavior
        if self.state == "SEEKING_GOAL":
            has_front_obstacle, front_obstacles = self.detect_front_obstacle()
            
            if has_front_obstacle:
                # Obstacle detected in front, start strafing away
                self.strafe_direction, side_obstacles = self.analyze_strafe_options()
                self.state = "STRAFING_AWAY"
                self.get_logger().info(f"Front obstacle detected, strafing {'LEFT' if self.strafe_direction == 1 else 'RIGHT'}")
                cmd = self.execute_strafing_away()
            else:
                # No front obstacle, continue seeking goal
                cmd = self.execute_goal_seeking(distance_to_goal)
                
        elif self.state == "STRAFING_AWAY":
            if self.is_front_clear():
                # Front is clear, start moving beside obstacle
                self.state = "MOVING_BESIDE"
                self.obstacle_cleared_position = (self.current_x, self.current_y)
                self.get_logger().info("Front clear, now moving beside obstacle")
                cmd = self.execute_moving_beside()
            else:
                # Continue strafing away from obstacle
                cmd = self.execute_strafing_away()
                
        elif self.state == "MOVING_BESIDE":
            if self.is_side_clear():
                # Side obstacle is cleared, start strafing back to path
                self.state = "STRAFING_BACK"
                self.get_logger().info("Side obstacle cleared, strafing back to path")
                cmd = self.execute_strafing_back()
            else:
                # Continue moving beside obstacle
                cmd = self.execute_moving_beside()
                
        elif self.state == "STRAFING_BACK":
            angle_to_goal = abs(self.get_angle_to_goal())
            
            # If we're reasonably aligned with goal or detect new front obstacle, return to goal seeking
            if angle_to_goal < math.radians(15) or self.detect_front_obstacle()[0]:
                self.state = "SEEKING_GOAL"
                self.strafe_direction = 0
                self.obstacle_cleared_position = None
                self.get_logger().info("Returning to goal seeking mode")
                cmd = self.execute_goal_seeking(distance_to_goal)
            else:
                # Continue strafing back toward path
                cmd = self.execute_strafing_back()
       
        # Publish command
        self.cmd_pub.publish(cmd)

    def execute_goal_seeking(self, distance_to_goal):
        """Execute goal seeking behavior"""
        cmd = Twist()
        angle_to_goal = self.get_angle_to_goal()
        
        # Angular control
        cmd.angular.z = angle_to_goal * 1.0
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))
        
        # Linear control based on alignment
        if abs(angle_to_goal) > math.radians(45):
            cmd.linear.x = self.max_linear_speed * 0.4
        elif abs(angle_to_goal) > math.radians(20):
            cmd.linear.x = self.max_linear_speed * 0.7
        else:
            cmd.linear.x = self.max_linear_speed
        
        self.get_logger().info(f"Seeking goal: dist={distance_to_goal:.2f}m, angle_diff={math.degrees(angle_to_goal):.1f}°")
        return cmd

    def execute_strafing_away(self):
        """Execute strafing away from obstacle (no forward movement)"""
        cmd = Twist()
        cmd.linear.x = 0.0  # No forward movement
        cmd.linear.y = self.strafe_direction * self.max_strafe_speed
        cmd.angular.z = 0.0  # No rotation while strafing
        
        direction_name = "LEFT" if self.strafe_direction == 1 else "RIGHT"
        self.get_logger().info(f"Strafing {direction_name} away from obstacle")
        return cmd

    def execute_moving_beside(self):
        """Execute moving forward beside the obstacle"""
        cmd = Twist()
        cmd.linear.x = self.max_linear_speed * 0.8  # Move forward beside obstacle
        cmd.linear.y = 0.0  # No lateral movement
        cmd.angular.z = 0.0  # Keep straight while moving beside
        
        self.get_logger().info("Moving forward beside obstacle")
        return cmd

    def execute_strafing_back(self):
        """Execute strafing back toward the goal path"""
        cmd = Twist()
        angle_to_goal = self.get_angle_to_goal()
        
        # Strafe back toward goal path (opposite of original strafe direction)
        cmd.linear.x = self.max_linear_speed * 0.6  # Slow forward movement while strafing back
        cmd.linear.y = -self.strafe_direction * self.max_strafe_speed * 0.7  # Strafe back
        
        # Gentle orientation correction toward goal
        cmd.angular.z = angle_to_goal * 0.5
        cmd.angular.z = max(-0.5, min(0.5, cmd.angular.z))
        
        direction_name = "RIGHT" if self.strafe_direction == 1 else "LEFT"
        self.get_logger().info(f"Strafing {direction_name} back to path")
        return cmd

    def publish_stop_command(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
   
    try:
        node = StrafingObstacleAvoidanceNavigation()
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
