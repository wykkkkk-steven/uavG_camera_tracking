import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageSnapshotNode(Node):
    def __init__(self):
        super().__init__('image_snapshot_node')

        self.image_topic = "/camera/image_raw"

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.saved = False
        self.get_logger().info(f'Subscribing to: {self.image_topic}')

    def image_callback(self, msg):
        if self.saved:
            return

        self.get_logger().info(
            f'Image received: {msg.width}x{msg.height}, '
            f'encoding={msg.encoding}, step={msg.step}'
        )

        if msg.encoding.lower() != 'rgb8':
            self.get_logger().error(
                f'Only rgb8 is supported by this simple saver, got {msg.encoding}'
            )
            return

        filename = '/tmp/camera_snapshot.ppm'

        # PPM image format: very simple RGB image file
        with open(filename, 'wb') as f:
            header = f'P6\n{msg.width} {msg.height}\n255\n'
            f.write(header.encode('ascii'))

            # Use msg.step in case each row has padding
            for y in range(msg.height):
                row_start = y * msg.step
                row_end = row_start + msg.width * 3
                f.write(bytes(msg.data[row_start:row_end]))

        self.saved = True
        self.get_logger().info(f'Saved image to {filename}')
        self.get_logger().info('Open it with: xdg-open /tmp/camera_snapshot.ppm')


def main(args=None):
    rclpy.init(args=args)
    node = ImageSnapshotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
