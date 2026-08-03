import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraTestNode(Node):
    def __init__(self):
        super().__init__('camera_test_node')

        self.image_topic = (
            '/world/default/model/x500_mono_cam_0/'
            'link/camera_link/sensor/imager/image'
        )

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

        # 每 10 帧打印一次，避免 terminal 刷太快
        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f'Image received: {msg.width} x {msg.height}, '
                f'encoding={msg.encoding}, frame={self.frame_count}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = CameraTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
