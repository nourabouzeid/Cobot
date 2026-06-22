#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Joints MoveIt2 controls — all others are passed through from Isaac
MOVEIT_JOINTS = {
    'joint2_to_joint1',
    'joint3_to_joint2',
    'joint4_to_joint3',
    'joint5_to_joint4',
    'joint6_to_joint5',
    'joint6output_to_joint6',
    'gripper_controller',
    'gripper_base_to_gripper_left2',
    'gripper_base_to_gripper_right3',
    'gripper_base_to_gripper_right2',
    'gripper_left3_to_gripper_left1',
    'gripper_right3_to_gripper_right1'
}

class IsaacDirectBridge(Node):
    def __init__(self):
        super().__init__('isaac_direct_bridge',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time',
                    rclpy.parameter.Parameter.Type.BOOL, True)
            ])

        # Latest known state of ALL joints from Isaac (includes wheels)
        self.isaac_joint_state = {}

        # Subscribe to Isaac's full joint states to cache wheel positions
        self.create_subscription(
            JointState,
            '/isaac_joint_states',
            self.isaac_callback,
            10)

        # Subscribe to MoveIt2's commanded positions
        self.create_subscription(
            JointState,
            '/joint_states',  # now remapped from /joint_states
            self.moveit_callback,
            10)

        self.pub = self.create_publisher(JointState, '/isaac_joint_command', 10)
        self.get_logger().info("Isaac bridge active — merging arm + wheel joints")

    def isaac_callback(self, msg):
        # Cache every joint Isaac reports (wheels + arm)
        for name, pos in zip(msg.name, msg.position):
            self.isaac_joint_state[name] = pos

    def moveit_callback(self, msg):
        if not msg.position or not self.isaac_joint_state:
            return

        # Start from the full cached Isaac state
        merged = dict(self.isaac_joint_state)

        # Overwrite only the joints MoveIt2 controls
        for name, pos in zip(msg.name, msg.position):
            if name in MOVEIT_JOINTS:
                merged[name] = pos

        # Build the outgoing message with all joints
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(merged.keys())
        out.position = list(merged.values())
        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacDirectBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
