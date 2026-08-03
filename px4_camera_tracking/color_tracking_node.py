import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class ColorTrackingNode(Node):
    def __init__(self):
        super().__init__('color_tracking_node')

        self.image_topic = "/camera/image_raw"
        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.frame_count = 0
        self.get_logger().info(f'Subscribing to: {self.image_topic}')

    def image_callback(self, msg):
        self.frame_count += 1

        if msg.encoding.lower() != 'rgb8':
            if self.frame_count % 10 == 0:
                self.get_logger().warn(f'Unsupported encoding: {msg.encoding}')
            return

        width = msg.width
        height = msg.height
        data = msg.data

        red_pixels = 0
        sum_x = 0
        sum_y = 0

        # 每隔 2 个像素采样一次
        step = 2

        for y in range(0, height, step):
            row_start = y * width * 3

            for x in range(0, width, step):
                i = row_start + x * 3

                r = data[i]
                g = data[i + 1]
                b = data[i + 2]

                # 红色判断：
                # 红色通道明显高，绿色和蓝色较低
                if r > 150 and g < 100 and b < 100:
                    red_pixels += 1
                    sum_x += x
                    sum_y += y

        if red_pixels == 0:
            if self.frame_count % 10 == 0:
                self.get_logger().info('Target not found')
            return

        target_cx = sum_x / red_pixels
        target_cy = sum_y / red_pixels

        image_cx = width / 2.0
        image_cy = height / 2.0

        error_x = target_cx - image_cx
        error_y = target_cy - image_cy

        # red_pixels 越大，红色目标在画面里占比越大
        area_estimate = red_pixels * step * step

        if self.frame_count % 5 == 0:
            self.get_logger().info(
                f'Target found | '
                f'center=({target_cx:.1f}, {target_cy:.1f}) | '
                f'area_est={area_estimate} | '
                f'error_x={error_x:.1f}, error_y={error_y:.1f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ColorTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
