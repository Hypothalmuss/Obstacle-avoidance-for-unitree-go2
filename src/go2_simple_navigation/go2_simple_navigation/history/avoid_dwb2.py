#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
import math
import numpy as np
from typing import List, Tuple, Optional

class DWBObstacleAvoidance(Node):
    """
    Dynamic Window-Based Obstacle Avoidance Node
    
    The DWB algorithm works by:
    1. Generating a set of possible velocity commands (linear and angular)
    2. Simulating robot trajectories for each velocity command
    3. Scoring each trajectory based on multiple criteria:
       - Obstacle distance (safety)
       - Goal alignment (efficiency)
       - Path smoothness (comfort)
    4. Selecting the best scoring trajectory
    """
    
    def __init__(self):
        super().__init__('dwb_obstacle_avoidance')
        
        # Robot physical parameters
        self.declare_parameter('robot_radius', 1.0)  # Robot's safety radius
        self.declare_parameter('max_linear_vel', 20.0)  # Maximum forward velocity
        self.declare_parameter('max_angular_vel', 20.0)  # Maximum angular velocity
        self.declare_parameter('linear_acceleration', 10.0)  # Linear acceleration limit
        self.declare_parameter('angular_acceleration', 20.0)  # Angular acceleration limit
        
        # DWB algorithm parameters
        self.declare_parameter('prediction_time', 2.0)  # How far ahead to predict (seconds)
        self.declare_parameter('dt', 0.1)  # Time step for trajectory simulation
        self.declare_parameter('linear_samples', 10)  # Number of linear velocity samples
        self.declare_parameter('angular_samples', 20)  # Number of angular velocity samples
        
        # Scoring weights (these determine behavior priority)
        self.declare_parameter('obstacle_weight', 5.0)  # Weight for obstacle avoidance
        self.declare_parameter('goal_weight', 2.0)     # Weight for goal alignment
        self.declare_parameter('smoothness_weight', 1.0)  # Weight for smooth motion
        
        # Detection parameters
        self.declare_parameter('detection_distance', 5.0)
        self.declare_parameter('min_height', -0.5)
        self.declare_parameter('max_height', 2.0)
        
        # Topic parameters
        self.declare_parameter('lidar_topic', '/velodyne_points')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('goal_topic', '/goal_point')  # Optional goal topic
        
        # Get all parameters
        self.robot_radius = self.get_parameter('robot_radius').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.linear_accel = self.get_parameter('linear_acceleration').value
        self.angular_accel = self.get_parameter('angular_acceleration').value
        
        self.prediction_time = self.get_parameter('prediction_time').value
        self.dt = self.get_parameter('dt').value
        self.linear_samples = self.get_parameter('linear_samples').value
        self.angular_samples = self.get_parameter('angular_samples').value
        
        self.obstacle_weight = self.get_parameter('obstacle_weight').value
        self.goal_weight = self.get_parameter('goal_weight').value
        self.smoothness_weight = self.get_parameter('smoothness_weight').value
        
        self.detection_distance = self.get_parameter('detection_distance').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        
        # Topic names
        lidar_topic = self.get_parameter('lidar_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        goal_topic = self.get_parameter('goal_topic').value
        
        # State variables
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.obstacle_points = []  # List of obstacle points from LiDAR
        self.goal_point = Point()  # Target goal (default: move forward)
        self.goal_point.x = 10.0  # Default goal 10 meters ahead
        self.goal_point.y = 0.0
        
        # Publishers and Subscribers
        self.subscription = self.create_subscription(
            PointCloud2, 
            lidar_topic, 
            self.cloud_callback, 
            10
        )
        
        self.goal_subscription = self.create_subscription(
            Point,
            goal_topic,
            self.goal_callback,
            10
        )
        
        self.publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("DWB Obstacle Avoidance Node Started")
        self.get_logger().info(f"Robot radius: {self.robot_radius}m")
        self.get_logger().info(f"Max velocities: {self.max_linear_vel} m/s, {self.max_angular_vel} rad/s")
        self.get_logger().info(f"Prediction time: {self.prediction_time}s")

    def goal_callback(self, msg: Point):
        """Update the goal point for navigation"""
        self.goal_point = msg
        self.get_logger().info(f"New goal received: ({msg.x:.2f}, {msg.y:.2f})")

    def cloud_callback(self, msg: PointCloud2):
        """
        Process LiDAR point cloud to extract obstacle points
        """
        self.obstacle_points = []
        
        try:
            for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                x, y, z = point
                
                # Filter points by height (remove ground and ceiling)
                if z < self.min_height or z > self.max_height:
                    continue
                
                # Only consider points within detection range
                distance = math.sqrt(x**2 + y**2)
                if distance > self.detection_distance:
                    continue
                
                # Store obstacle point (x, y coordinates)
                self.obstacle_points.append((x, y))
                
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")

    def generate_velocity_samples(self) -> List[Tuple[float, float]]:
        """
        Generate velocity samples within the dynamic window
        
        The dynamic window is the set of velocities reachable within one time step
        given the current velocity and acceleration constraints.
        
        Returns:
            List of (linear_vel, angular_vel) tuples
        """
        velocities = []
        
        # Calculate velocity bounds based on current velocity and acceleration limits
        min_linear = max(-self.max_linear_vel, 
                        self.current_linear_vel - self.linear_accel * self.dt)
        max_linear = min(self.max_linear_vel, 
                        self.current_linear_vel + self.linear_accel * self.dt)
        
        min_angular = max(-self.max_angular_vel, 
                         self.current_angular_vel - self.angular_accel * self.dt)
        max_angular = min(self.max_angular_vel, 
                         self.current_angular_vel + self.angular_accel * self.dt)
        
        # Generate samples within the dynamic window
        linear_step = (max_linear - min_linear) / max(1, self.linear_samples - 1)
        angular_step = (max_angular - min_angular) / max(1, self.angular_samples - 1)
        
        for i in range(self.linear_samples):
            linear_vel = min_linear + i * linear_step
            for j in range(self.angular_samples):
                angular_vel = min_angular + j * angular_step
                velocities.append((linear_vel, angular_vel))
        
        return velocities

    def simulate_trajectory(self, linear_vel: float, angular_vel: float) -> List[Tuple[float, float, float]]:
        """
        Simulate robot trajectory for given velocities
        
        Args:
            linear_vel: Linear velocity (m/s)
            angular_vel: Angular velocity (rad/s)
            
        Returns:
            List of (x, y, theta) poses along the trajectory
        """
        trajectory = []
        x, y, theta = 0.0, 0.0, 0.0  # Start from robot's current pose
        
        # Simulate trajectory for prediction_time duration
        num_steps = int(self.prediction_time / self.dt)
        
        for _ in range(num_steps):
            # Update pose using bicycle model
            x += linear_vel * math.cos(theta) * self.dt
            y += linear_vel * math.sin(theta) * self.dt
            theta += angular_vel * self.dt
            
            trajectory.append((x, y, theta))
        
        return trajectory

    def check_collision(self, trajectory: List[Tuple[float, float, float]]) -> bool:
        """
        Check if trajectory collides with obstacles
        
        Args:
            trajectory: List of (x, y, theta) poses
            
        Returns:
            True if collision detected, False otherwise
        """
        for x, y, _ in trajectory:
            # Check collision with each obstacle point
            for obs_x, obs_y in self.obstacle_points:
                distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                if distance < self.robot_radius:
                    return True
        return False

    def score_obstacle_distance(self, trajectory: List[Tuple[float, float, float]]) -> float:
        """
        Score trajectory based on distance to closest obstacle
        Higher score = safer trajectory
        """
        if not self.obstacle_points:
            return 1.0  # No obstacles, maximum score
        
        min_distance = float('inf')
        
        for x, y, _ in trajectory:
            for obs_x, obs_y in self.obstacle_points:
                distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                min_distance = min(min_distance, distance)
        
        # Normalize score (closer to obstacle = lower score)
        if min_distance < self.robot_radius:
            return 0.0  # Collision
        else:
            # Score decreases exponentially as we get closer to obstacles
            return min(1.0, min_distance / (2.0 * self.robot_radius))

    def score_goal_alignment(self, trajectory: List[Tuple[float, float, float]]) -> float:
        """
        Score trajectory based on alignment with goal
        Higher score = better goal alignment
        """
        if not trajectory:
            return 0.0
        
        # Get final position of trajectory
        final_x, final_y, final_theta = trajectory[-1]
        
        # Calculate angle to goal from final position
        goal_angle = math.atan2(self.goal_point.y - final_y, 
                               self.goal_point.x - final_x)
        
        # Calculate alignment score based on final heading vs goal direction
        angle_diff = abs(final_theta - goal_angle)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)  # Normalize to [0, π]
        
        # Score decreases with angle difference
        alignment_score = 1.0 - (angle_diff / math.pi)
        
        # Also consider distance progress toward goal
        goal_distance = math.sqrt((final_x - self.goal_point.x)**2 + 
                                 (final_y - self.goal_point.y)**2)
        initial_goal_distance = math.sqrt(self.goal_point.x**2 + self.goal_point.y**2)
        
        progress_score = max(0.0, (initial_goal_distance - goal_distance) / initial_goal_distance)
        
        return 0.7 * alignment_score + 0.3 * progress_score

    def score_smoothness(self, linear_vel: float, angular_vel: float) -> float:
        """
        Score trajectory based on smoothness (continuity with current velocity)
        Higher score = smoother motion
        """
        linear_diff = abs(linear_vel - self.current_linear_vel)
        angular_diff = abs(angular_vel - self.current_angular_vel)
        
        # Normalize differences
        linear_smooth = 1.0 - min(1.0, linear_diff / self.max_linear_vel)
        angular_smooth = 1.0 - min(1.0, angular_diff / self.max_angular_vel)
        
        return 0.5 * linear_smooth + 0.5 * angular_smooth

    def evaluate_trajectory(self, linear_vel: float, angular_vel: float) -> float:
        """
        Comprehensive trajectory evaluation combining all scoring criteria
        
        Returns:
            Combined score (higher = better trajectory)
        """
        # Simulate trajectory
        trajectory = self.simulate_trajectory(linear_vel, angular_vel)
        
        # Check for collision (immediate disqualification)
        if self.check_collision(trajectory):
            return -1.0  # Invalid trajectory
        
        # Calculate individual scores
        obstacle_score = self.score_obstacle_distance(trajectory)
        goal_score = self.score_goal_alignment(trajectory)
        smoothness_score = self.score_smoothness(linear_vel, angular_vel)
        
        # Combine scores with weights
        total_score = (self.obstacle_weight * obstacle_score + 
                      self.goal_weight * goal_score + 
                      self.smoothness_weight * smoothness_score)
        
        return total_score

    def select_best_velocity(self) -> Tuple[float, float]:
        """
        Select the best velocity command using DWB algorithm
        
        Returns:
            Tuple of (best_linear_vel, best_angular_vel)
        """
        # Generate velocity samples
        velocity_samples = self.generate_velocity_samples()
        
        best_score = -float('inf')
        best_linear_vel = 0.0
        best_angular_vel = 0.0
        
        # Evaluate each velocity sample
        for linear_vel, angular_vel in velocity_samples:
            score = self.evaluate_trajectory(linear_vel, angular_vel)
            
            if score > best_score:
                best_score = score
                best_linear_vel = linear_vel
                best_angular_vel = angular_vel
        
        # If no valid trajectory found, stop the robot
        if best_score < 0:
            self.get_logger().warn("No safe trajectory found! Stopping robot.")
            return 0.0, 0.0
        
        return best_linear_vel, best_angular_vel

    def control_loop(self):
        """
        Main control loop - runs DWB algorithm and publishes velocity commands
        """
        # Select best velocity using DWB algorithm
        linear_vel, angular_vel = self.select_best_velocity()
        
        # Update current velocity for next iteration
        self.current_linear_vel = linear_vel
        self.current_angular_vel = angular_vel
        
        # Create and publish velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel
        
        self.publisher.publish(cmd_vel)
        
        # Optional: Log status
        if len(self.obstacle_points) > 0:
            self.get_logger().info_throttle(
                2.0,  # Log every 2 seconds
                f"DWB: v={linear_vel:.2f} m/s, ω={angular_vel:.2f} rad/s, "
                f"obstacles={len(self.obstacle_points)}"
            )

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DWBObstacleAvoidance()
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
