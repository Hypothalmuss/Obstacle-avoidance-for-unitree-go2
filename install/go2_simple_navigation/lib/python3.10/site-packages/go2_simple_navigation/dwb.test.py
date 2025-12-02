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

class DWBObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('dwb_obstacle_avoidance')
        
        # Navigation parameters
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tolerance', 0.02)
        
        # DWB Parameters - Restored normal forward motion
        self.declare_parameter('max_linear_vel', 1.2)  # Good forward speed
        self.declare_parameter('min_linear_vel', -0.2)  # Limited backing up
        self.declare_parameter('max_angular_vel', 1.2)  # Good turning speed
        self.declare_parameter('min_angular_vel', -1.2)
        
        # Acceleration limits (for dynamic window)
        self.declare_parameter('max_linear_acc', 1.0)  # Reduced from 2.0
        self.declare_parameter('max_angular_acc', 2.0)  # Reduced from 3.0
        
        # Trajectory evaluation parameters
        self.declare_parameter('prediction_time', 2.0)  # Restored original
        self.declare_parameter('time_step', 0.1)        # Restored finer resolution
        self.declare_parameter('v_resolution', 0.1)     # Restored original
        self.declare_parameter('w_resolution', 0.1)     # Angular velocity resolution
        
        # Cost function weights - Better balanced for normal navigation
        self.declare_parameter('obstacle_cost_weight', 5.0)   # Strong but not overwhelming
        self.declare_parameter('goal_cost_weight', 2.0)       # Important goal seeking
        self.declare_parameter('velocity_cost_weight', 1.0)   # Prefer forward motion
        self.declare_parameter('smoothness_cost_weight', 0.3) # Some smoothness
        
        # Obstacle detection parameters
        self.declare_parameter('detection_distance', 3.0)  # Reduced from 4.0
        self.declare_parameter('robot_radius', 0.35)       # Slightly increased
        self.declare_parameter('min_height', -0.3)
        self.declare_parameter('max_height', 2.0)
        self.declare_parameter('safety_margin', 0.2)       # Added safety margin
        
        # Get parameters
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.min_linear_vel = self.get_parameter('min_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.min_angular_vel = self.get_parameter('min_angular_vel').value
        
        self.max_linear_acc = self.get_parameter('max_linear_acc').value
        self.max_angular_acc = self.get_parameter('max_angular_acc').value
        
        self.prediction_time = self.get_parameter('prediction_time').value
        self.time_step = self.get_parameter('time_step').value
        self.v_resolution = self.get_parameter('v_resolution').value
        self.w_resolution = self.get_parameter('w_resolution').value
        
        self.obstacle_weight = self.get_parameter('obstacle_cost_weight').value
        self.goal_weight = self.get_parameter('goal_cost_weight').value
        self.velocity_weight = self.get_parameter('velocity_cost_weight').value
        self.smoothness_weight = self.get_parameter('smoothness_cost_weight').value
        
        self.detection_distance = self.get_parameter('detection_distance').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.safety_margin = self.get_parameter('safety_margin').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        
        # Topic parameters
        self.declare_parameter('lidar_topic', '/velodyne_points')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        
        lidar_topic = self.get_parameter('lidar_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        
        # Subscribers and Publishers
        self.lidar_sub = self.create_subscription(PointCloud2, lidar_topic, self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        
        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # State variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.obstacles = []  # List of obstacle points [(x, y)]
        self.goal_reached = False
        
        # Starting position
        self.start_x = None
        self.start_y = None
        
        # Previous command for smoothness
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0
        
        # Stuck detection - More lenient
        self.stuck_counter = 0
        self.max_stuck_time = 30  # 3 seconds at 10Hz - longer before considering stuck
        self.last_position = (0.0, 0.0)
        self.position_threshold = 0.02  # 2cm movement threshold - smaller threshold
        
        self.get_logger().info("DWB Navigation node started")
        self.get_logger().info(f"Goal: ({self.goal_x}, {self.goal_y}) meters from start")
        self.get_logger().info(f"Prediction time: {self.prediction_time}s")

    def odom_callback(self, msg):
        """Update current robot state from odometry"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Get current velocities
        self.current_linear_vel = msg.twist.twist.linear.x
        self.current_angular_vel = msg.twist.twist.angular.z
        
        # Convert quaternion to yaw
        orientation = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        # Set starting position
        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.last_position = (self.current_x, self.current_y)
            self.get_logger().info(f"Starting position: ({self.start_x:.2f}, {self.start_y:.2f})")

    def lidar_callback(self, msg):
        """Process lidar data to extract obstacle points"""
        self.obstacles = []
        
        try:
            for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                x, y, z = point
                
                # Filter by height
                if z < self.min_height or z > self.max_height:
                    continue
                
                # Check if within detection range (keep points behind for backing up)
                distance = math.sqrt(x**2 + y**2)
                if distance <= self.detection_distance:
                    # Transform to global coordinates
                    global_x = self.current_x + x * math.cos(self.current_yaw) - y * math.sin(self.current_yaw)
                    global_y = self.current_y + x * math.sin(self.current_yaw) + y * math.cos(self.current_yaw)
                    self.obstacles.append((global_x, global_y))
                    
        except Exception as e:
            self.get_logger().error(f"Error processing lidar: {e}")

    def get_dynamic_window(self):
        """Calculate the dynamic window of achievable velocities"""
        dt = self.time_step
        
        # Current velocity constraints
        min_v = max(self.min_linear_vel, self.current_linear_vel - self.max_linear_acc * dt)
        max_v = min(self.max_linear_vel, self.current_linear_vel + self.max_linear_acc * dt)
        
        min_w = max(self.min_angular_vel, self.current_angular_vel - self.max_angular_acc * dt)
        max_w = min(self.max_angular_vel, self.current_angular_vel + self.max_angular_acc * dt)
        
        return min_v, max_v, min_w, max_w

    def predict_trajectory(self, v, w):
        """Predict robot trajectory for given velocities"""
        trajectory = []
        x, y, theta = self.current_x, self.current_y, self.current_yaw
        
        num_steps = int(self.prediction_time / self.time_step)
        
        for i in range(num_steps):
            # Simple motion model (constant velocity)
            if abs(w) < 1e-6:  # Straight line motion
                x += v * self.time_step * math.cos(theta)
                y += v * self.time_step * math.sin(theta)
            else:  # Circular motion
                # Instantaneous center of rotation
                R = v / w
                cx = x - R * math.sin(theta)
                cy = y + R * math.cos(theta)
                
                # Update position
                theta += w * self.time_step
                x = cx + R * math.sin(theta)
                y = cy - R * math.cos(theta)
            
            trajectory.append((x, y, theta))
        
        return trajectory

    def is_trajectory_safe(self, trajectory):
        """Check if trajectory is collision-free"""
        if not self.obstacles:
            return True
        
        safe_distance = self.robot_radius + self.safety_margin
        
        for traj_point in trajectory:
            traj_x, traj_y, _ = traj_point
            
            for obs_x, obs_y in self.obstacles:
                distance = math.sqrt((traj_x - obs_x)**2 + (traj_y - obs_y)**2)
                if distance < safe_distance:
                    return False
        
        return True

    def calculate_obstacle_cost(self, trajectory):
        """Calculate obstacle cost for a trajectory - Only penalize when obstacles exist"""
        if not self.obstacles:
            return 0.0  # No cost if no obstacles
        
        # First check if trajectory is safe
        if not self.is_trajectory_safe(trajectory):
            return 1000.0  # Invalid trajectory
        
        min_distance = float('inf')
        
        for traj_point in trajectory:
            traj_x, traj_y, _ = traj_point
            
            for obs_x, obs_y in self.obstacles:
                distance = math.sqrt((traj_x - obs_x)**2 + (traj_y - obs_y)**2)
                min_distance = min(min_distance, distance)
        
        # Only penalize if actually close to obstacles
        if min_distance < 1.5:  # Only penalize when getting close
            # Smooth cost function for gradual avoidance
            return 1.0 / (min_distance + 0.2)
        return 0.0

    def calculate_goal_cost(self, trajectory):
        """Calculate goal alignment cost - Improved version"""
        if not trajectory:
            return 1000.0
        
        # Global goal position
        goal_global_x = self.start_x + self.goal_x if self.start_x else self.goal_x
        goal_global_y = self.start_y + self.goal_y if self.start_y else self.goal_y
        
        # Calculate progress toward goal
        current_dist_to_goal = math.sqrt((goal_global_x - self.current_x)**2 + 
                                       (goal_global_y - self.current_y)**2)
        
        # Use trajectory endpoint
        final_x, final_y, final_theta = trajectory[-1]
        final_dist_to_goal = math.sqrt((goal_global_x - final_x)**2 + 
                                     (goal_global_y - final_y)**2)
        
        # Prefer trajectories that make progress toward goal
        progress_cost = final_dist_to_goal
        
        # Add heading cost only if close to goal
        heading_cost = 0.0
        if current_dist_to_goal < 1.0:
            goal_angle = math.atan2(goal_global_y - final_y, goal_global_x - final_x)
            angle_diff = abs(goal_angle - final_theta)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)
            heading_cost = 0.5 * angle_diff
        
        return progress_cost + heading_cost

    def calculate_velocity_cost(self, v, w):
        """Strongly prefer forward motion, discourage backing up unless necessary"""
        if v > 0:
            # Reward higher forward speeds
            v_cost = (self.max_linear_vel - v) / self.max_linear_vel * 0.5
        elif v == 0:
            # Neutral cost for stopping
            v_cost = 0.5
        else:
            # High cost for backing up (only use when necessary)
            v_cost = 2.0 + abs(v) / abs(self.min_linear_vel)
        
        # Prefer lower angular velocities for straight motion
        w_cost = abs(w) / self.max_angular_vel * 0.5
        return v_cost + w_cost

    def calculate_smoothness_cost(self, v, w):
        """Penalize large changes in velocity"""
        dv = abs(v - self.prev_linear_vel)
        dw = abs(w - self.prev_angular_vel)
        
        # Normalize by maximum possible change
        dv_norm = dv / (self.max_linear_acc * self.time_step + 1e-6)
        dw_norm = dw / (self.max_angular_acc * self.time_step + 1e-6)
        
        return dv_norm + dw_norm

    def evaluate_trajectory(self, v, w):
        """Evaluate a single velocity pair"""
        trajectory = self.predict_trajectory(v, w)
        
        # Calculate individual costs
        obstacle_cost = self.calculate_obstacle_cost(trajectory)
        goal_cost = self.calculate_goal_cost(trajectory)
        velocity_cost = self.calculate_velocity_cost(v, w)
        smoothness_cost = self.calculate_smoothness_cost(v, w)
        
        # Weighted total cost
        total_cost = (self.obstacle_weight * obstacle_cost +
                     self.goal_weight * goal_cost +
                     self.velocity_weight * velocity_cost +
                     self.smoothness_weight * smoothness_cost)
        
        return total_cost, obstacle_cost, goal_cost

    def is_stuck(self):
        """Detect if robot is stuck - only when obstacles are present"""
        # Don't consider stuck if no obstacles nearby
        if len(self.obstacles) < 10:  # Very few or no obstacles
            self.stuck_counter = 0
            return False
            
        current_pos = (self.current_x, self.current_y)
        distance_moved = math.sqrt((current_pos[0] - self.last_position[0])**2 + 
                                 (current_pos[1] - self.last_position[1])**2)
        
        # Also check if velocities are very low
        low_velocity = abs(self.current_linear_vel) < 0.05 and abs(self.current_angular_vel) < 0.1
        
        if distance_moved < self.position_threshold and low_velocity:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = current_pos
        
        return self.stuck_counter > self.max_stuck_time

    def get_escape_velocities(self):
        """Generate escape velocities when stuck"""
        escape_velocities = []
        
        # Try backing up with different angular velocities
        for w in np.arange(-0.5, 0.6, 0.1):
            escape_velocities.append((-0.2, w))  # Back up slowly
        
        # Try pure rotation
        for w in np.arange(-1.0, 1.1, 0.2):
            if abs(w) > 0.1:  # Avoid zero angular velocity
                escape_velocities.append((0.0, w))
        
        return escape_velocities

    def dwb_planning(self):
        """Main DWB planning algorithm - Improved version"""
        min_v, max_v, min_w, max_w = self.get_dynamic_window()
        
        best_v, best_w = 0.0, 0.0
        best_cost = float('inf')
        best_obstacle_cost = 0.0
        best_goal_cost = 0.0
        
        # Check if stuck and need escape maneuver
        stuck = self.is_stuck()
        if stuck:
            self.get_logger().warn("Robot appears stuck, trying escape maneuvers")
            velocity_pairs = self.get_escape_velocities()
        else:
            # Normal velocity sampling
            v_samples = np.arange(min_v, max_v + self.v_resolution, self.v_resolution)
            w_samples = np.arange(min_w, max_w + self.w_resolution, self.w_resolution)
            velocity_pairs = [(v, w) for v in v_samples for w in w_samples]
        
        evaluated_trajectories = 0
        valid_trajectories = 0
        
        for v, w in velocity_pairs:
            # Skip if velocities are outside dynamic window (for escape maneuvers)
            if not stuck:
                if v < min_v or v > max_v or w < min_w or w > max_w:
                    continue
            
            total_cost, obstacle_cost, goal_cost = self.evaluate_trajectory(v, w)
            evaluated_trajectories += 1
            
            # Only consider valid (non-collision) trajectories
            if obstacle_cost < 999.0:  # Valid trajectory
                valid_trajectories += 1
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_v = v
                    best_w = w
                    best_obstacle_cost = obstacle_cost
                    best_goal_cost = goal_cost
        
        # If no valid trajectory found, emergency stop
        if valid_trajectories == 0:
            self.get_logger().warn("No valid trajectory found! Emergency stop.")
            best_v, best_w = 0.0, 0.0
        
        self.get_logger().info(f"DWB: Evaluated {evaluated_trajectories} trajectories, "
                             f"Valid: {valid_trajectories}, Stuck: {stuck}, "
                             f"Best: v={best_v:.2f}, w={best_w:.2f}")
        
        return best_v, best_w

    def control_loop(self):
        """Main control loop"""
        if self.start_x is None:
            return
        
        # Check if goal reached
        goal_global_x = self.start_x + self.goal_x
        goal_global_y = self.start_y + self.goal_y
        distance_to_goal = math.sqrt((goal_global_x - self.current_x)**2 + 
                                   (goal_global_y - self.current_y)**2)
        
        if distance_to_goal < self.goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info("Goal reached!")
                self.goal_reached = True
            self.publish_stop_command()
            return
        
        # Run DWB planning
        linear_vel, angular_vel = self.dwb_planning()
        
        # Update previous velocities for smoothness calculation
        self.prev_linear_vel = linear_vel
        self.prev_angular_vel = angular_vel
        
        # Publish command
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_pub.publish(cmd)
        
        self.get_logger().info(f"Command: v={linear_vel:.2f}, w={angular_vel:.2f}, "
                             f"Goal distance: {distance_to_goal:.2f}m, "
                             f"Obstacles: {len(self.obstacles)}")

    def publish_stop_command(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

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
