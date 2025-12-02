import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import math
import struct
from sensor_msgs_py import point_cloud2

class VFHNode(Node):
    def __init__(self):
        super().__init__('vfh_node')
        
        # ROS2 publishers and subscribers
        # Changed from /scan to /velodyne_points
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, 
            '/velodyne_points', 
            self.pointcloud_callback, 
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # VFH parameters
        self.ranges = []
        self.angles = []
        self.max_range = 10.0
        
        # VFH configuration
        self.alpha = 5.0  # Bin width in degrees
        self.threshold = 2.0  # Threshold for obstacle detection
        self.safety_distance = 0.5  # Minimum safe distance from obstacles
        self.max_linear_speed = 0.3
        self.max_angular_speed = 1.0
        
        # Point cloud processing parameters
        self.min_height = -0.5  # Minimum height to consider (relative to sensor)
        self.max_height = 2.0   # Maximum height to consider
        self.min_range = 0.1    # Minimum range to consider
        
        # Target direction (0 = forward)
        self.target_direction = 0.0
        
        self.get_logger().info('VFH Node initialized for point cloud data')
    
    def pointcloud_callback(self, msg):
        """Process incoming point cloud data and convert to 2D laser scan"""
        try:
            # Extract points from point cloud
            points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            
            if not points:
                self.ranges = []
                self.angles = []
                return
            
            # Handle different point cloud data structures
            if len(points) == 0:
                self.ranges = []
                self.angles = []
                return
            
            # Check if points are tuples or have different structure
            x_coords = []
            y_coords = []
            z_coords = []
            
            for point in points:
                if isinstance(point, (list, tuple)):
                    if len(point) >= 3:
                        x_coords.append(point[0])
                        y_coords.append(point[1])
                        z_coords.append(point[2])
                else:
                    # Handle structured array or other formats
                    try:
                        x_coords.append(float(point.x if hasattr(point, 'x') else point[0]))
                        y_coords.append(float(point.y if hasattr(point, 'y') else point[1]))
                        z_coords.append(float(point.z if hasattr(point, 'z') else point[2]))
                    except:
                        continue
            
            if not x_coords:
                self.ranges = []
                self.angles = []
                return
            
            # Convert to numpy arrays
            x = np.array(x_coords)
            y = np.array(y_coords)
            z = np.array(z_coords)
            
            # Filter points by height (keep only points at robot level)
            valid_height_mask = (z >= self.min_height) & (z <= self.max_height)
            x_filtered = x[valid_height_mask]
            y_filtered = y[valid_height_mask]
            
            if len(x_filtered) == 0:
                self.ranges = []
                self.angles = []
                return
            
            # Calculate ranges and angles
            ranges = np.sqrt(x_filtered**2 + y_filtered**2)
            angles = np.arctan2(y_filtered, x_filtered)
            
            # Filter by range
            valid_range_mask = (ranges >= self.min_range) & (ranges <= self.max_range)
            self.ranges = ranges[valid_range_mask]
            self.angles = angles[valid_range_mask]
            
            # Remove invalid readings
            valid_mask = np.isfinite(self.ranges) & np.isfinite(self.angles)
            self.ranges = self.ranges[valid_mask]
            self.angles = self.angles[valid_mask]
            
            # Log debug info occasionally
            if len(self.ranges) > 0:
                self.get_logger().debug(f'Processed {len(self.ranges)} valid points from point cloud')
            
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {str(e)}')
            # Log more details for debugging
            self.get_logger().error(f'Point cloud info - width: {msg.width}, height: {msg.height}, fields: {[f.name for f in msg.fields]}')
            self.ranges = []
            self.angles = []
    
    def create_polar_histogram(self):
        """Create polar histogram from point cloud data"""
        if len(self.ranges) == 0:
            return None, None
            
        # Define histogram bins (-180 to 180 degrees)
        num_bins = int(360 / self.alpha)
        bin_angles = np.linspace(-180, 180, num_bins, endpoint=False)
        histogram = np.zeros(num_bins)
        
        # Process each point
        for i, (distance, angle) in enumerate(zip(self.ranges, self.angles)):
            if distance <= self.min_range:  # Invalid reading
                continue
                
            # Convert angle to degrees
            angle_deg = math.degrees(angle)
            
            # Normalize angle to [-180, 180]
            while angle_deg > 180:
                angle_deg -= 360
            while angle_deg <= -180:
                angle_deg += 360
            
            # Find corresponding histogram bin
            bin_index = int((angle_deg + 180) / self.alpha) % num_bins
            
            # Calculate obstacle density (closer obstacles have higher density)
            if distance < self.safety_distance:
                density = 10.0  # Very high density for close obstacles
            else:
                # Inverse square law with distance weighting
                density = max(0, (2.0 - distance) ** 2) if distance < 2.0 else 0.0
            
            histogram[bin_index] = max(histogram[bin_index], density)
        
        return histogram, bin_angles
    
    def smooth_histogram(self, histogram):
        """Apply smoothing to reduce noise in histogram"""
        if histogram is None:
            return None
            
        # Apply Gaussian-like smoothing
        kernel = np.array([0.25, 0.5, 0.25])
        smoothed = np.convolve(histogram, kernel, mode='same')
        
        # Handle circular boundary
        smoothed[0] = 0.25 * histogram[-1] + 0.5 * histogram[0] + 0.25 * histogram[1]
        smoothed[-1] = 0.25 * histogram[-2] + 0.5 * histogram[-1] + 0.25 * histogram[0]
        
        return smoothed
    
    def find_valleys(self, histogram, bin_angles):
        """Find valleys (safe directions) in the histogram"""
        if histogram is None:
            return []
            
        valleys = []
        in_valley = False
        valley_start = 0
        
        for i, density in enumerate(histogram):
            if density < self.threshold and not in_valley:
                # Start of a valley
                in_valley = True
                valley_start = i
            elif density >= self.threshold and in_valley:
                # End of a valley
                in_valley = False
                valley_end = i - 1
                valley_center = (valley_start + valley_end) // 2
                valley_width = valley_end - valley_start + 1
                valleys.append({
                    'center_index': valley_center,
                    'center_angle': bin_angles[valley_center],
                    'width': valley_width,
                    'start': valley_start,
                    'end': valley_end
                })
        
        # Handle valley that wraps around
        if in_valley:
            valley_end = len(histogram) - 1
            valley_center = (valley_start + valley_end) // 2
            valley_width = valley_end - valley_start + 1
            valleys.append({
                'center_index': valley_center,
                'center_angle': bin_angles[valley_center],
                'width': valley_width,
                'start': valley_start,
                'end': valley_end
            })
        
        return valleys
    
    def select_direction(self, valleys):
        """Select the best direction from available valleys"""
        if not valleys:
            return None
            
        # Score valleys based on alignment with target and width
        best_valley = None
        best_score = -float('inf')
        
        for valley in valleys:
            # Angular difference from target direction
            angle_diff = abs(valley['center_angle'] - self.target_direction)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Score: prefer wider valleys and directions closer to target
            width_score = valley['width'] * 0.1
            direction_score = max(0, 90 - angle_diff) / 90.0
            total_score = width_score + direction_score * 2.0
            
            if total_score > best_score:
                best_score = total_score
                best_valley = valley
        
        return best_valley
    
    def control_loop(self):
        """Main control loop for VFH navigation"""
        if len(self.ranges) == 0:
            # No point cloud data available yet
            return
        
        try:
            # Create and smooth polar histogram
            histogram, bin_angles = self.create_polar_histogram()
            if histogram is None:
                return
                
            smoothed_histogram = self.smooth_histogram(histogram)
            
            # Find safe directions (valleys)
            valleys = self.find_valleys(smoothed_histogram, bin_angles)
            
            # Select best direction
            selected_valley = self.select_direction(valleys)
            
            # Create velocity command
            twist = Twist()
            
            if selected_valley is None:
                # No safe direction found - stop and rotate
                self.get_logger().warn('No safe direction found - stopping')
                twist.linear.x = 0.0
                twist.angular.z = self.max_angular_speed * 0.5  # Slow rotation to find path
            else:
                # Calculate steering angle
                steering_angle = math.radians(selected_valley['center_angle'])
                
                # Adjust linear speed based on obstacle proximity
                if len(self.ranges) > 0:
                    min_distance = np.min(self.ranges)
                    if min_distance < self.safety_distance:
                        speed_factor = max(0.1, min_distance / self.safety_distance)
                    else:
                        speed_factor = 1.0
                else:
                    speed_factor = 0.1
                
                # Reduce speed when turning
                turn_factor = max(0.3, 1.0 - abs(steering_angle) / math.pi)
                
                twist.linear.x = self.max_linear_speed * speed_factor * turn_factor
                twist.angular.z = np.clip(-steering_angle * 2.0, 
                                        -self.max_angular_speed, 
                                        self.max_angular_speed)
                
                # Log current action for debugging
                self.get_logger().debug(
                    f'Selected direction: {selected_valley["center_angle"]:.1f}°, '
                    f'Linear: {twist.linear.x:.2f}, Angular: {twist.angular.z:.2f}'
                )
            
            # Publish command
            self.cmd_pub.publish(twist)
            
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {str(e)}')
            # Publish stop command on error
            twist = Twist()
            self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VFHNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
