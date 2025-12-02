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

class WallFollowingStrafingNavigation(Node):
    def __init__(self):
        super().__init__('wall_following_strafing_navigation')
       
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tolerance', 0.05)
        self.declare_parameter('detection_distance', 3.0)
        self.declare_parameter('min_height', -0.3)
        self.declare_parameter('max_height', 2.0)
        self.declare_parameter('max_linear_speed', 2.2)
        self.declare_parameter('max_strafe_speed', 1.5)
        self.declare_parameter('safety_distance', 1.5)
        self.declare_parameter('wall_follow_distance', 1.0)  # Distance to maintain from wall
       
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.detection_distance = self.get_parameter('detection_distance').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_strafe_speed = self.get_parameter('max_strafe_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.wall_follow_distance = self.get_parameter('wall_follow_distance').value
       
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
       
        # Navigation states
        self.state = 'NORMAL'  # NORMAL, STRAFING_AWAY, WALL_FOLLOWING, STRAFING_BACK
        self.obstacle_side = None  # 'left' or 'right'
        self.original_path_offset = 0.0  # Track deviation from original path
        self.wall_follow_start_x = 0.0
        self.wall_follow_start_y = 0.0
       
        # Lidar sectors (in robot frame)
        self.lidar_sectors = {
            'front': [],
            'front_left': [],
            'front_right': [],
            'left': [],
            'right': [],
            'rear_left': [],
            'rear_right': [],
            'rear': []
        }
       
        self.get_logger().info(f"Wall-Following Strafing Navigation node started")
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
        """Process 360-degree lidar data into sectors"""
        # Reset sectors
        for sector in self.lidar_sectors:
            self.lidar_sectors[sector] = []
       
        self.obstacles = []
       
        try:
            for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                x, y, z = point
               
                if z < self.min_height or z > self.max_height:
                    continue
               
                distance = math.sqrt(x**2 + y**2)
                angle = math.atan2(y, x)  # -π to π
                angle_deg = math.degrees(angle)
               
                if distance <= self.detection_distance:
                    obstacle = (x, y, distance, angle)
                    self.obstacles.append(obstacle)
                   
                    # Classify into sectors
                    if -30 <= angle_deg <= 30:
                        self.lidar_sectors['front'].append(obstacle)
                    elif 30 < angle_deg <= 75:
                        self.lidar_sectors['front_left'].append(obstacle)
                    elif 75 < angle_deg <= 105:
                        self.lidar_sectors['left'].append(obstacle)
                    elif 105 < angle_deg <= 150:
                        self.lidar_sectors['rear_left'].append(obstacle)
                    elif 150 < angle_deg <= 180 or -180 <= angle_deg <= -150:
                        self.lidar_sectors['rear'].append(obstacle)
                    elif -150 < angle_deg <= -105:
                        self.lidar_sectors['rear_right'].append(obstacle)
                    elif -105 < angle_deg <= -75:
                        self.lidar_sectors['right'].append(obstacle)
                    elif -75 < angle_deg < -30:
                        self.lidar_sectors['front_right'].append(obstacle)
           
        except Exception as e:
            self.get_logger().error(f"Error processing lidar: {e}")

    def get_sector_min_distance(self, sector_name):
        """Get minimum distance in a sector"""
        if not self.lidar_sectors[sector_name]:
            return float('inf')
        return min(obs[2] for obs in self.lidar_sectors[sector_name])

    def is_sector_clear(self, sector_name, min_distance=None):
        """Check if a sector is clear of obstacles"""
        if min_distance is None:
            min_distance = self.safety_distance
        return self.get_sector_min_distance(sector_name) > min_distance

    def calculate_goal_angle(self):
        """Calculate angle to goal from current position"""
        goal_global_x = self.start_x + self.goal_x
        goal_global_y = self.start_y + self.goal_y
        goal_angle = math.atan2(goal_global_y - self.current_y, goal_global_x - self.current_x)
       
        angle_diff = goal_angle - self.current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
           
        return goal_angle, angle_diff

    def is_diagonal_path_clear(self):
        """Check if diagonal path back to original route is clear"""
        if self.obstacle_side == 'left':
            # Check front_right and right sectors for return path
            return (self.is_sector_clear('front_right') and
                   self.is_sector_clear('right', self.wall_follow_distance))
        else:  # obstacle_side == 'right'
            # Check front_left and left sectors for return path
            return (self.is_sector_clear('front_left') and
                   self.is_sector_clear('left', self.wall_follow_distance))

    def has_cleared_obstacle(self):
        """Check if we've moved past the obstacle"""
        if self.obstacle_side == 'left':
            # Obstacle was on left, check if rear_left is clear
            return self.is_sector_clear('rear_left', self.wall_follow_distance)
        else:  # obstacle_side == 'right'
            # Obstacle was on right, check if rear_right is clear
            return self.is_sector_clear('rear_right', self.wall_follow_distance)

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
       
        # State machine logic
        if self.state == 'NORMAL':
            cmd = self.normal_navigation()
           
        elif self.state == 'STRAFING_AWAY':
            cmd = self.strafe_away_from_obstacle()
           
        elif self.state == 'WALL_FOLLOWING':
            cmd = self.follow_wall()
           
        elif self.state == 'STRAFING_BACK':
            cmd = self.strafe_back_to_path()
       
        # Publish command
        self.cmd_pub.publish(cmd)

    def normal_navigation(self):
        """Normal goal-seeking behavior"""
        cmd = Twist()
       
        # Check for obstacles ahead
        front_clear = self.is_sector_clear('front')
        front_left_clear = self.is_sector_clear('front_left')
        front_right_clear = self.is_sector_clear('front_right')
       
        if not front_clear:
            # Need to avoid obstacle - decide which side to strafe to
            left_clear = self.is_sector_clear('left')
            right_clear = self.is_sector_clear('right')
           
            if left_clear and (not right_clear or front_left_clear):
                # Strafe left
                self.state = 'STRAFING_AWAY'
                self.obstacle_side = 'right'  # Obstacle will be on right side
                self.get_logger().info("Obstacle ahead - starting to strafe LEFT")
               
            elif right_clear:
                # Strafe right
                self.state = 'STRAFING_AWAY'
                self.obstacle_side = 'left'  # Obstacle will be on left side
                self.get_logger().info("Obstacle ahead - starting to strafe RIGHT")
               
            else:
                # Both sides blocked - stop and back up
                cmd.linear.x = -0.3
                self.get_logger().info("All paths blocked - backing up")
                return cmd
       
        # Normal goal seeking
        goal_angle, angle_diff = self.calculate_goal_angle()
       
        cmd.angular.z = angle_diff * 1.5
        cmd.angular.z = max(-self.max_strafe_speed, min(self.max_strafe_speed, cmd.angular.z))
       
        if abs(angle_diff) > math.radians(30):
            cmd.linear.x = self.max_linear_speed * 0.3
        else:
            cmd.linear.x = self.max_linear_speed
           
        return cmd

    def strafe_away_from_obstacle(self):
        """Strafe sideways to avoid obstacle until it's beside us"""
        cmd = Twist()
       
        if self.obstacle_side == 'right':
            # Strafing left, check if obstacle is now on our right side
            if (not self.is_sector_clear('right', self.wall_follow_distance) and
                self.is_sector_clear('front')):
                # Obstacle is beside us, start wall following
                self.state = 'WALL_FOLLOWING'
                self.wall_follow_start_x = self.current_x
                self.wall_follow_start_y = self.current_y
                self.get_logger().info("Obstacle now on RIGHT side - starting wall following")
            else:
                # Continue strafing left
                cmd.linear.y = self.max_strafe_speed
                cmd.linear.x = self.max_linear_speed * 0.3
               
        else:  # obstacle_side == 'left'
            # Strafing right, check if obstacle is now on our left side
            if (not self.is_sector_clear('left', self.wall_follow_distance) and
                self.is_sector_clear('front')):
                # Obstacle is beside us, start wall following
                self.state = 'WALL_FOLLOWING'
                self.wall_follow_start_x = self.current_x
                self.wall_follow_start_y = self.current_y
                self.get_logger().info("Obstacle now on LEFT side - starting wall following")
            else:
                # Continue strafing right
                cmd.linear.y = -self.max_strafe_speed
                cmd.linear.x = self.max_linear_speed * 0.3
       
        return cmd

    def follow_wall(self):
        """Follow alongside the obstacle while maintaining safe distance"""
        cmd = Twist()
       
        # Check if we've cleared the obstacle and can return to path
        if self.has_cleared_obstacle() and self.is_diagonal_path_clear():
            self.state = 'STRAFING_BACK'
            self.get_logger().info("Obstacle cleared - starting to strafe back to original path")
            return self.strafe_back_to_path()
       
        # Move forward while maintaining distance from wall
        cmd.linear.x = self.max_linear_speed * 0.8
       
        # Adjust lateral position to maintain wall distance
        if self.obstacle_side == 'right':
            right_distance = self.get_sector_min_distance('right')
            if right_distance < self.wall_follow_distance:
                # Too close to right wall, strafe left slightly
                cmd.linear.y = 0.3
            elif right_distance > self.wall_follow_distance * 1.5:
                # Too far from right wall, strafe right slightly
                cmd.linear.y = -0.2
               
        else:  # obstacle_side == 'left'
            left_distance = self.get_sector_min_distance('left')
            if left_distance < self.wall_follow_distance:
                # Too close to left wall, strafe right slightly
                cmd.linear.y = -0.3
            elif left_distance > self.wall_follow_distance * 1.5:
                # Too far from left wall, strafe left slightly
                cmd.linear.y = 0.2
       
        # Gentle orientation correction toward goal
        _, angle_diff = self.calculate_goal_angle()
        cmd.angular.z = angle_diff * 0.3
       
        return cmd

    def strafe_back_to_path(self):
        """Strafe back toward original path"""
        cmd = Twist()
       
        # Check if we're back on path or close enough
        # This is simplified - in practice you'd track the exact original path
        goal_global_x = self.start_x + self.goal_x
        goal_global_y = self.start_y + self.goal_y
       
        # Calculate if we're roughly back on the direct line to goal
        # For simplicity, check if we can go straight to goal
        if self.is_sector_clear('front') and abs(self.calculate_goal_angle()[1]) < math.radians(15):
            self.state = 'NORMAL'
            self.obstacle_side = None
            self.get_logger().info("Back on path - resuming normal navigation")
            return self.normal_navigation()
       
        # Continue strafing back
        if self.obstacle_side == 'right':
            # We were avoiding obstacle on right, so strafe right to get back
            cmd.linear.y = -self.max_strafe_speed * 0.7
        else:  # obstacle_side == 'left'
            # We were avoiding obstacle on left, so strafe left to get back
            cmd.linear.y = self.max_strafe_speed * 0.7
           
        cmd.linear.x = self.max_linear_speed * 0.5
       
        # Orientation correction toward goal
        _, angle_diff = self.calculate_goal_angle()
        cmd.angular.z = angle_diff * 0.5
       
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
        node = WallFollowingStrafingNavigation()
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
