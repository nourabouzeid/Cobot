#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, WorkspaceParameters,
    Constraints, PositionConstraint, OrientationConstraint,
    JointConstraint, BoundingVolume
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped

from scipy.spatial.transform import Rotation as R

# TF2 imports for frame transformation
import tf2_ros
import tf2_geometry_msgs  # registers PoseStamped transform support
import time


# ---------------------------------------------------------------------------
# Gripper states → joint angle (degrees → radians)
# Physical range: -42 (fully closed) … 9 (fully open)
# ---------------------------------------------------------------------------
GRIPPER_ANGLES_DEG = {
    'open':    9.0,
    'neutral': 0.0,
    'close':  -42.0,
}


class EndEffectorCommander(Node):
    def __init__(self):
        super().__init__('end_effector_commander',
                         parameter_overrides=[
                             rclpy.parameter.Parameter('use_sim_time',
                                 rclpy.parameter.Parameter.Type.BOOL, True)
                         ])
        self._action_client = ActionClient(self, MoveGroup, '/move_action')

        # TF2 buffer and listener
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=True)

        self.get_logger().info('Node started, waiting for /move_action action server...')

    # ------------------------------------------------------------------
    # Gripper control
    # ------------------------------------------------------------------

    def move_gripper(self, state: str):
        """
        Move the gripper to one of three named states:
            'open'    →  9° (fully open)
            'neutral' →  0°
            'close'   → -42° (fully closed)

        Args:
            state: one of 'open', 'neutral', 'close'
        """
        if state not in GRIPPER_ANGLES_DEG:
            self.get_logger().error(
                f"Unknown gripper state '{state}'. "
                f"Choose from: {list(GRIPPER_ANGLES_DEG.keys())}"
            )
            return

        angle_deg = GRIPPER_ANGLES_DEG[state]
        angle_rad = math.radians(angle_deg)

        self._action_client.wait_for_server()

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'gripper'

        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5

        joint_constraint = JointConstraint()
        joint_constraint.joint_name = 'gripper_controller'
        joint_constraint.position = angle_rad
        joint_constraint.tolerance_above = 0.01
        joint_constraint.tolerance_below = 0.01
        joint_constraint.weight = 1.0

        constraints = Constraints()
        constraints.joint_constraints.append(joint_constraint)
        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options.plan_only = False

        self.get_logger().info(
            f"Moving gripper → '{state}' ({angle_deg}° / {angle_rad:.4f} rad)"
        )

        future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    # ------------------------------------------------------------------
    # Arm / end-effector control  (unchanged from original)
    # ------------------------------------------------------------------

    def transform_pose_to_g_base(self, x, y, z, rx, ry, rz):
        """
        Transforms only the position from 'map' to 'g_base'.
        Orientation is kept as-is in g_base (not transformed).
        """
        r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        qx, qy, qz, qw = r.as_quat()

        point_in_map = PoseStamped()
        point_in_map.header.frame_id = 'map'
        point_in_map.header.stamp = self.get_clock().now().to_msg()
        point_in_map.pose.position.x = x
        point_in_map.pose.position.y = y
        point_in_map.pose.position.z = z
        point_in_map.pose.orientation.w = 1.0  # identity, ignored after transform

        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame='g_base',
                source_frame='map',
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=3.0)
            )
            transformed = tf2_geometry_msgs.do_transform_pose(point_in_map.pose, transform)

            final_pose = Pose()
            final_pose.position = transformed.position
            final_pose.orientation.x = qx
            final_pose.orientation.y = qy
            final_pose.orientation.z = qz
            final_pose.orientation.w = qw

            self.get_logger().info(
                f'Transformed map({x:.3f}, {y:.3f}, {z:.3f}) -> '
                f'g_base({final_pose.position.x:.3f}, '
                f'{final_pose.position.y:.3f}, '
                f'{final_pose.position.z:.3f})'
            )
            return final_pose

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF transform failed: {e}')
            raise

    def send_goal(self, x, y, z, rx=-90.0, ry=0.0, rz=-90.0, frame='map'):
        """
        Sends a motion plan request for the arm.
        x, y, z are in 'frame' (default: 'map').
        Orientation rx/ry/rz are in degrees (euler XYZ).
        """
        self._action_client.wait_for_server()
        self.get_logger().info('/move_action action server found')

        if frame != 'g_base':
            target_pose = self.transform_pose_to_g_base(x, y, z, rx, ry, rz)
        else:
            r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
            qx, qy, qz, qw = r.as_quat()
            target_pose = Pose()
            target_pose.position.x = x
            target_pose.position.y = y
            target_pose.position.z = z
            target_pose.orientation.x = qx
            target_pose.orientation.y = qy
            target_pose.orientation.z = qz
            target_pose.orientation.w = qw

        final_pose = [
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w,
        ]
        print(final_pose)

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'arm_group'

        req.workspace_parameters = WorkspaceParameters()
        req.workspace_parameters.header.frame_id = 'g_base'
        req.workspace_parameters.min_corner.x = -0.5
        req.workspace_parameters.min_corner.y = -0.5
        req.workspace_parameters.min_corner.z = -0.5
        req.workspace_parameters.max_corner.x = 0.5
        req.workspace_parameters.max_corner.y = 0.5
        req.workspace_parameters.max_corner.z = 0.5

        req.num_planning_attempts = 10
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = 'g_base'
        pos_constraint.link_name = 'tool_tip'
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.0001]
        bv = BoundingVolume()
        bv.primitives.append(primitive)
        bv.primitive_poses.append(target_pose)
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0

        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = 'g_base'
        ori_constraint.link_name = 'tool_tip'
        ori_constraint.orientation = target_pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.1
        ori_constraint.absolute_y_axis_tolerance = 0.1
        ori_constraint.absolute_z_axis_tolerance = 0.1
        ori_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)
        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options.plan_only = False

        self.get_logger().info(
            f'Sending goal in g_base: position=({target_pose.position.x:.3f}, '
            f'{target_pose.position.y:.3f}, {target_pose.position.z:.3f})'
        )

        send_goal_future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    # ------------------------------------------------------------------
    # Shared callbacks
    # ------------------------------------------------------------------

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by move_action')
            return
        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        state = feedback_msg.feedback.state
        self.get_logger().info(f'Feedback: {state}')

    def result_callback(self, future):
        result = future.result().result
        error_code = result.error_code.val
        if error_code == 1:
            self.get_logger().info('SUCCESS: motion executed')
        else:
            self.get_logger().error(f'FAILED: error_code={error_code}')


def main(args=None):
    rclpy.init(args=args)
    node = EndEffectorCommander()
    time.sleep(2.0)

    # --- Arm movement example ---
    # node.send_goal(x=0.24, y=0.0, z=0.2, rx=-70, ry=0.0, rz=0.0, frame='map')

    # --- Gripper examples ---
    # node.move_gripper('open')      # → +9°   fully open
    # node.move_gripper('neutral')   # →  0°   neutral
    node.move_gripper('close')     # → -42°  fully closed

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
