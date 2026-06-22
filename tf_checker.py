#!/usr/bin/env python3
"""
TF Frame Coordinate Transformer
Usage:
    python3 tf_transform.py <source_frame> <target_frame> <x> <y> <z>

Example:
    python3 tf_transform.py map g_base 0.26022 0.01832 0.25
"""

import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
import time


class FrameTransformer(Node):
    def __init__(self):
        super().__init__('frame_transformer')
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=True)

    def transform(self, source_frame, target_frame, x, y, z):
        # Wait for TF buffer to populate
        time.sleep(1.5)

        pose_in = PoseStamped()
        pose_in.header.frame_id = source_frame
        pose_in.header.stamp = self.get_clock().now().to_msg()
        pose_in.pose.position.x = x
        pose_in.pose.position.y = y
        pose_in.pose.position.z = z
        pose_in.pose.orientation.w = 1.0  # identity

        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=5.0)
            )
            pose_out = tf2_geometry_msgs.do_transform_pose(pose_in.pose, transform)

            print(f'\nInput  [{source_frame}]: x={x}, y={y}, z={z}')
            print(f'Output [{target_frame}]: '
                  f'x={pose_out.position.x:.6f}, '
                  f'y={pose_out.position.y:.6f}, '
                  f'z={pose_out.position.z:.6f}\n')

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            print(f'\n[ERROR] Transform failed: {e}')
            print(f'  Make sure both frames "{source_frame}" and "{target_frame}" exist.')
            print(f'  Check available frames with: ros2 run tf2_tools view_frames\n')
            sys.exit(1)


def main():
    if len(sys.argv) != 6:
        print('Usage: python3 tf_transform.py <source_frame> <target_frame> <x> <y> <z>')
        print('Example: python3 tf_transform.py map g_base 0.26022 0.01832 0.25')
        sys.exit(1)

    source_frame = sys.argv[1]
    target_frame = sys.argv[2]
    x = float(sys.argv[3])
    y = float(sys.argv[4])
    z = float(sys.argv[5])

    rclpy.init()
    node = FrameTransformer()
    node.transform(source_frame, target_frame, x, y, z)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
