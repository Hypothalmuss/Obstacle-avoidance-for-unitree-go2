import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class WalkForward(Node):
    def __init__(self):
        super().__init__('walk_forward')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        timer_period = 0.1  # publish every 100ms
        self.timer = self.create_timer(timer_period, self.publish_cmd)

    def publish_cmd(self):
        msg = Twist()
        msg.linear.x = 1000.0  # linear speed
        msg.angular.z = 0.0  # no rotation
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: linear.x={msg.linear.x}')

def main(args=None):
    rclpy.init(args=args)
    node = WalkForward()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

