#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class ScanObstacleStop(Node):
    def __init__(self):
        super().__init__('scan_obstacle_stop')
        
        # Parameters
        self.declare_parameter('detection_distance', 1.5)
        self.declare_parameter('detection_angle', 100.0)   # total FOV
        self.declare_parameter('forward_speed', 1.0)
        self.declare_parameter('lidar_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.detection_distance = self.get_parameter('detection_distance').value
        self.detection_angle = self.get_parameter('detection_angle').value / 2.0
        self.forward_speed = self.get_parameter('forward_speed').value

        lidar_topic = self.get_parameter('lidar_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        
        # Sub + Pub
        self.subscription = self.create_subscription(
            LaserScan, lidar_topic, self.scan_callback, 10
        )
        self.publisher = self.create_publisher(Twist, cmd_vel_topic, 10)

        # Timer for sending cmd_vel
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)

        # State
        self.obstacle_detected = False
        self.closest_obstacle_distance = float('inf')

        self.get_logger().info("Obstacle stop node using /scan started")

    def scan_callback(self, msg: LaserScan):
        self.obstacle_detected = False
        self.closest_obstacle_distance = float('inf')

        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # Loop over scan ranges
        for i, distance in enumerate(msg.ranges):
            if math.isinf(distance) or math.isnan(distance):
                continue

            # Convert index to angle in degrees
            angle = math.degrees(angle_min + i * angle_increment)

            # Only front half cone: |angle| < detection_angle
            if abs(angle) > self.detection_angle:
                continue

            # Check detection distance
            if distance < self.detection_distance:
                self.obstacle_detected = True
                if distance < self.closest_obstacle_distance:
                    self.closest_obstacle_distance = distance

        if self.obstacle_detected:
            self.get_logger().info(
                f"Obstacle detected at {self.closest_obstacle_distance:.2f} m"
            )

    def publish_cmd_vel(self):
        msg = Twist()

        if self.obstacle_detected:
            msg.linear.x = 0.0
        else:
            msg.linear.x = self.forward_speed
        
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanObstacleStop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

