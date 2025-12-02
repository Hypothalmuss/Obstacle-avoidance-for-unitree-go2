#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
import sensor_msgs_py.point_cloud2 as pc2
import math

class VelodyneObstacleStop(Node):
    def __init__(self):
        super().__init__('velodyne_obstacle_stop')
        

        self.declare_parameter('detection_distance', 1.5)  
        self.declare_parameter('detection_angle', 100.0)    
        self.declare_parameter('forward_speed', 1.0)       
        self.declare_parameter('min_height', -0.5)         
        self.declare_parameter('max_height', 2.0)          
        
        # Get parameters
        self.detection_distance = self.get_parameter('detection_distance').value
        self.detection_angle = self.get_parameter('detection_angle').value / 2.0  
        self.forward_speed = self.get_parameter('forward_speed').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        
        # Topic names - make them configurable
        self.declare_parameter('lidar_topic', '/velodyne_points')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        lidar_topic = self.get_parameter('lidar_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        
        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            PointCloud2, 
            lidar_topic, 
            self.cloud_callback, 
            10
        )
        self.publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        
        # Timer for publishing commands
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)
        
        # State variables
        self.obstacle_detected = False
        self.closest_obstacle_distance = float('inf')
        self.obstacle_count = 0
        
        self.get_logger().info(f"Obstacle detection node started")
        self.get_logger().info(f"Detection distance: {self.detection_distance}m")
        self.get_logger().info(f"Detection angle: ±{self.detection_angle}°")
        self.get_logger().info(f"Forward speed: {self.forward_speed} m/s")

    def cloud_callback(self, msg):
        self.obstacle_detected = False
        self.closest_obstacle_distance = float('inf')
        self.obstacle_count = 0
        
        try:
            for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                x, y, z = point
                
                # Skip points that are too high or too low (likely ground/ceiling)
                if z < self.min_height or z > self.max_height:
                    continue
                
                # Only consider points in front of the robot
                if x <= 0:
                    continue
                
                distance = math.sqrt(x**2 + y**2)  # 2D distance (ignore height)
                
                # Check if point is within detection cone
                angle = math.degrees(math.atan2(abs(y), x))
                
                if angle <= self.detection_angle and distance < self.detection_distance:
                    self.obstacle_detected = True
                    self.obstacle_count += 1
                    if distance < self.closest_obstacle_distance:
                        self.closest_obstacle_distance = distance
                        
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")
            # In case of error, assume obstacle detected for safety
            self.obstacle_detected = True
        
        # Log detection info (but not too frequently)
        if self.obstacle_detected and self.obstacle_count > 5:  # Filter noise
            self.get_logger().info(
                f"Obstacle detected! Closest: {self.closest_obstacle_distance:.2f}m, "
                f"Points: {self.obstacle_count}"
            )

    def publish_cmd_vel(self):
        msg = Twist()
        
        if self.obstacle_detected and self.obstacle_count > 1:  # Add threshold to filter noise
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            # Optional: add a small backward movement for safety
            # msg.linear.x = -0.1
        else:
            msg.linear.x = self.forward_speed
            msg.angular.z = 0.0
            
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VelodyneObstacleStop()
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
