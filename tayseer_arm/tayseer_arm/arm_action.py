#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, WorkspaceParameters,
    Constraints, PositionConstraint, OrientationConstraint,
    JointConstraint, BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped

from scipy.spatial.transform import Rotation as R

import tf2_ros
import tf2_geometry_msgs

from tayseer_interfaces.action import ArmManipulation


# ---------------------------------------------------------------------------
# Gripper states → joint angle (degrees → radians)
# Physical range: -42 (fully closed) … 9 (fully open)
# ---------------------------------------------------------------------------
GRIPPER_ANGLES_DEG = {
    'open':    9.0,
    'neutral': 0.0,
    'close':  -42.0,
}

# Grasp direction → (rx, ry, rz) in degrees
# "Top"   — gripper descends from above
# "Front" — gripper approaches horizontally from the front
GRASP_ORIENTATIONS = {
    'Top':   (-80.0,  10.0, -90.0),
    'Front': (  0.0,  10.0, -90.0),
}

# How long to wait above the object before closing (seconds)
HOVER_SECONDS = 1.0

# How much to lift after grasping (metres, in g_base z)
LIFT_Z_OFFSET = 0.05


class ArmManipulationServer(Node):
    """
    Action server that exposes /arm_manipulate (tayseer_interfaces/ArmManipulation).

    Pick sequence  (pick=True)
    --------------
    1. Open gripper
    2. Transform object position from 'map' → 'g_base'
    3. Move arm to target pose
    4. Close gripper
    5. Lift arm by LIFT_Z_OFFSET in z
    6. Return success / failure

    Place sequence (pick=False)
    ---------------
    1. Transform target position from 'map' → 'g_base'
    2. Move arm to target pose
    3. Open gripper
    4. Lift arm by LIFT_Z_OFFSET in z
    5. Return success / failure

    IMPORTANT: every step that waits on a MoveGroup result is implemented
    with `await` on the future directly. Do NOT replace these with
    `rclpy.spin_until_future_complete(self, ...)` — calling a blocking spin
    function on this node from inside a callback that this node's own
    executor is already running breaks the executor's ability to deliver
    any subsequent action goal (the goal just silently never arrives).
    This is a known rclpy limitation, not a bug in this script's logic.
    """

    def __init__(self):
        super().__init__(
            'arm_manipulation_server',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.parameter.Parameter.Type.BOOL,
                    True,
                )
            ],
        )

        self._cb_group = ReentrantCallbackGroup()

        # MoveIt action client
        self._move_client = ActionClient(
            self, MoveGroup, '/move_action',
            callback_group=self._cb_group,
        )

        # TF2
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer, self, spin_thread=True
        )

        # Arm manipulation action server
        self._action_server = ActionServer(
            self,
            ArmManipulation,
            '/arm_manipulate',
            execute_callback=self._execute,
            callback_group=self._cb_group,
        )

        self.get_logger().info('ArmManipulationServer ready on /arm_manipulate')

    # ------------------------------------------------------------------
    # Action server execute callback (coroutine)
    # ------------------------------------------------------------------

    async def _execute(self, goal_handle: ServerGoalHandle):
        obj_name       = goal_handle.request.object_name
        pos            = goal_handle.request.object_position   # float32[3] in 'map'
        grasp_direction = goal_handle.request.grasp_direction
        orientation = goal_handle.request.orientation
        is_pick        = goal_handle.request.pick

        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        rx, ry, rz = float(orientation[0]), float(orientation[1]), float(orientation[2])

        mode = 'pick' if is_pick else 'place'

        self.get_logger().info(
            f"[{mode}] Starting {mode} of '{obj_name}' at map({x:.3f}, {y:.3f}, {z:.3f}) "
            f"grasp='{grasp_direction}'"
        )

        if grasp_direction in GRASP_ORIENTATIONS:
            rx, ry, rz = GRASP_ORIENTATIONS[grasp_direction]

        # ---- Stage 0: open gripper (pick only) --------------------------
        if is_pick:
            self._publish_feedback(goal_handle, 'opening_gripper', 0.0)
            success, msg = await self._move_gripper('open')
            if not success:
                return self._abort(goal_handle, f'Gripper open failed: {msg}')

        # ---- Stage 1: approach ------------------------------------------
        self._publish_feedback(goal_handle, 'approaching', 0.2)

        try:
            target_pose = self._transform_to_g_base(x, y, z, rx=rx, ry=ry, rz=rz)
        except Exception as e:
            return self._abort(goal_handle, f'TF transform failed: {e}')

        success, msg = await self._move_arm(target_pose)
        if not success:
            return self._abort(goal_handle, f'Arm move failed: {msg}')

        self._publish_feedback(goal_handle, 'grasping' if is_pick else 'placing', 0.5)
        time.sleep(HOVER_SECONDS)

        # ---- Stage 2: gripper action ------------------------------------
        # Pick  → close gripper
        # Place → open gripper
        gripper_state = 'close' if is_pick else 'open'
        success, msg = await self._move_gripper(gripper_state)
        if not success:
            return self._abort(goal_handle, f'Gripper {gripper_state} failed: {msg}')

        # ---- Stage 3: lift ----------------------------------------------
        if is_pick:
            self._publish_feedback(goal_handle, 'lifting', 0.8)

            lift_pose = Pose()
            lift_pose.position.x = target_pose.position.x
            lift_pose.position.y = target_pose.position.y
            lift_pose.position.z = target_pose.position.z + LIFT_Z_OFFSET
            lift_pose.orientation = target_pose.orientation

            success, msg = await self._move_arm(lift_pose)
            if not success:
                return self._abort(goal_handle, f'Lift move failed: {msg}')

        self._publish_feedback(goal_handle, 'done', 1.0)

        # ---- Done -------------------------------------------------------
        result = ArmManipulation.Result()
        result.success = True
        result.message = f"{mode.capitalize()}ed '{obj_name}' successfully"
        goal_handle.succeed()
        self.get_logger().info(f"[{mode}] {result.message}")
        return result

    # ------------------------------------------------------------------
    # MoveIt helpers — coroutines, awaiting futures directly
    # ------------------------------------------------------------------

    async def _send_moveit_goal(self, moveit_goal: MoveGroup.Goal):
        """
        Sends a MoveGroup goal and awaits the result.
        Returns (error_code: int, error_string: str).
        """
        self._move_client.wait_for_server()

        send_goal_handle = await self._move_client.send_goal_async(moveit_goal)
        if not send_goal_handle.accepted:
            return -1, 'Goal rejected by MoveIt'

        result_response = await send_goal_handle.get_result_async()
        error_code = result_response.result.error_code.val
        return error_code, ''

    async def _move_arm(self, target_pose: Pose):
        """Move arm to target_pose (already in g_base). Returns (success, msg)."""
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
        primitive.dimensions = [0.001]
        bv = BoundingVolume()
        bv.primitives.append(primitive)
        bv.primitive_poses.append(target_pose)
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0

        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = 'g_base'
        ori_constraint.link_name = 'tool_tip'
        ori_constraint.orientation = target_pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.2
        ori_constraint.absolute_y_axis_tolerance = 0.2
        ori_constraint.absolute_z_axis_tolerance = 0.2
        ori_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)
        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options.plan_only = False

        self.get_logger().info(
            f'[arm] Moving to g_base({target_pose.position.x:.3f}, '
            f'{target_pose.position.y:.3f}, {target_pose.position.z:.3f})'
        )

        code, msg = await self._send_moveit_goal(goal)
        if code == 1:
            self.get_logger().info('[arm] SUCCESS')
            return True, 'ok'
        else:
            self.get_logger().error(f'[arm] FAILED error_code={code}')
            return False, f'error_code={code}'

    async def _move_gripper(self, state: str):
        """Close/open gripper by joint constraint. Returns (success, msg)."""
        if state not in GRIPPER_ANGLES_DEG:
            return False, f"Unknown gripper state '{state}'"

        angle_rad = math.radians(GRIPPER_ANGLES_DEG[state])

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'gripper'
        req.num_planning_attempts = 20
        req.allowed_planning_time = 20.0
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5

        jc = JointConstraint()
        jc.joint_name = 'gripper_controller'
        jc.position = angle_rad
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0

        constraints = Constraints()
        constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options.plan_only = False

        self.get_logger().info(
            f'[gripper] Moving to {state} '
            f'({GRIPPER_ANGLES_DEG[state]}° / {angle_rad:.4f} rad)'
        )

        code, msg = await self._send_moveit_goal(goal)
        if code == 1:
            self.get_logger().info('[gripper] SUCCESS')
            return True, 'ok'
        else:
            self.get_logger().error(f'[gripper] FAILED error_code={code}')
            return False, f'error_code={code}'

    # ------------------------------------------------------------------
    # TF2 helper
    # ------------------------------------------------------------------

    def _transform_to_g_base(self, x, y, z, rx, ry, rz):
        """
        Transforms position from 'map' → 'g_base'.
        Orientation (rx/ry/rz degrees, XYZ euler) is applied in g_base directly.
        """
        r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        qx, qy, qz, qw = r.as_quat()

        pose_in_map = PoseStamped()
        pose_in_map.header.frame_id = 'map'
        pose_in_map.header.stamp = self.get_clock().now().to_msg()
        pose_in_map.pose.position.x = x
        pose_in_map.pose.position.y = y
        pose_in_map.pose.position.z = z
        pose_in_map.pose.orientation.w = 1.0  # identity; only position used

        transform = self._tf_buffer.lookup_transform(
            target_frame='g_base',
            source_frame='map',
            time=rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=3.0),
        )
        transformed = tf2_geometry_msgs.do_transform_pose(pose_in_map.pose, transform)

        pose = Pose()
        pose.position = transformed.position
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        self.get_logger().info(
            f'[tf] map({x:.3f}, {y:.3f}, {z:.3f}) → '
            f'g_base({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})'
        )
        return pose

    # ------------------------------------------------------------------
    # Feedback / abort helpers
    # ------------------------------------------------------------------

    def _publish_feedback(self, goal_handle: ServerGoalHandle, stage: str, progress: float):
        fb = ArmManipulation.Feedback()
        fb.stage = stage
        fb.progress = progress
        goal_handle.publish_feedback(fb)
        self.get_logger().info(f'[feedback] stage={stage} progress={progress:.0%}')

    def _abort(self, goal_handle: ServerGoalHandle, message: str):
        self.get_logger().error(f'[arm_manipulate] ABORT — {message}')
        result = ArmManipulation.Result()
        result.success = False
        result.message = message
        goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)

    executor = MultiThreadedExecutor()
    node = ArmManipulationServer()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()