#!/usr/bin/env python3
"""
gripper_depth_picker.py

Action server:  /gripper_pick  (type: tayseer_interfaces/action/ArmManipulation)

Workflow
--------
1. Receives goal with rough object position (map frame).
2. Moves arm to a SCAN pose offset from the object along the approach direction.
3. Captures one depth image from the gripper camera.
4. Refines the 3-D position using depth-only point-cloud clustering.
5. Calls /arm_manipulate with the refined position to perform the pick.
"""

import math
import time
import threading
from typing import Optional

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, Point, PointStamped
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener
import sensor_msgs.msg

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, WorkspaceParameters,
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from scipy.spatial.transform import Rotation as R

from tayseer_interfaces.action import ArmManipulation


# ---------------------------------------------------------------------------
# Grasp presets (must match your existing arm action server)
# ---------------------------------------------------------------------------
GRASP_ORIENTATIONS = {
    'Top':   (-80.0,  10.0, -90.0),
    'Front': (  0.0,  10.0, -90.0),
}


class GripperDepthPicker(Node):
    def __init__(self):
        super().__init__('gripper_depth_picker')

        self._cb_group = ReentrantCallbackGroup()

        # ── TF ───────────────────────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        # ── Action clients ───────────────────────────────────────────────
        self._move_client = ActionClient(
            self, MoveGroup, '/move_action', callback_group=self._cb_group
        )
        self._arm_client = ActionClient(
            self, ArmManipulation, '/arm_manipulate', callback_group=self._cb_group
        )

        # ── Joint states (for MoveIt start_state) ────────────────────────
        self._latest_joint_state: Optional = None
        self._joint_state_lock = threading.Lock()
        self.create_subscription(
            sensor_msgs.msg.JointState, '/joint_states', self._joint_state_callback, 10
        )

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('gripper_depth_topic', '/gripper/depth')
        self.declare_parameter('gripper_camera_info_topic', '/gripper/camera_info')
        self.declare_parameter('gripper_camera_frame', 'gripper_camera_link')
        self.declare_parameter('scan_distance', 0.25)      # m back from object
        self.declare_parameter('search_radius', 0.10)      # m around expected pos
        self.declare_parameter('min_cluster_points', 100)
        self.declare_parameter('depth_max', 2.0)

        # Intrinsics fallback (if CameraInfo never arrives)
        self.declare_parameter('fx', 0.0)
        self.declare_parameter('fy', 0.0)
        self.declare_parameter('cx', 0.0)
        self.declare_parameter('cy', 0.0)

        # ── Action server ────────────────────────────────────────────────
        # We reuse ArmManipulation action type; clients call /gripper_pick
        self._action_server = ActionServer(
            self,
            ArmManipulation,
            '/gripper_pick',
            execute_callback=self._execute,
            callback_group=self._cb_group,
        )

        # ── Depth subscriber ─────────────────────────────────────────────
        self._bridge = CvBridge()
        self._latest_depth: Optional[np.ndarray] = None
        self._depth_header = None
        self._depth_lock = threading.Lock()

        depth_topic = self.get_parameter('gripper_depth_topic').value
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, depth_topic, self._depth_callback, qos_profile=qos)

        # ── Camera info ────────────────────────────────────────────────────
        self._intrinsics_ready = False
        self._fx = self._fy = self._cx = self._cy = 0.0
        self._cam_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('gripper_camera_info_topic').value,
            self._cam_info_callback,
            10,
        )

        self.get_logger().info('GripperDepthPicker ready on /gripper_pick')

    def _joint_state_callback(self, msg):
        with self._joint_state_lock:
            self._latest_joint_state = msg

    # ------------------------------------------------------------------
    # Camera info
    # ------------------------------------------------------------------
    def _cam_info_callback(self, msg: CameraInfo):
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]
        self._intrinsics_ready = True
        self.destroy_subscription(self._cam_info_sub)
        self._cam_info_sub = None
        self.get_logger().info(
            f'Gripper intrinsics: fx={self._fx:.2f} fy={self._fy:.2f} '
            f'cx={self._cx:.2f} cy={self._cy:.2f}'
        )

    # ------------------------------------------------------------------
    # Depth image
    # ------------------------------------------------------------------
    def _depth_callback(self, msg: Image):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth decode failed: {e}')
            return

        if msg.encoding == '16UC1':
            depth = depth.astype(np.float32) / 1000.0
        elif msg.encoding == '32FC1':
            depth = depth.astype(np.float32)
        else:
            self.get_logger().warn(f'Unsupported depth encoding {msg.encoding}')
            return

        with self._depth_lock:
            self._latest_depth = depth
            self._depth_header = msg.header

    # ------------------------------------------------------------------
    # Main execute callback
    # ------------------------------------------------------------------
    async def _execute(self, goal_handle: ServerGoalHandle):
        obj_name = goal_handle.request.object_name
        rough_pos = goal_handle.request.object_position   # float32[3] in map
        grasp_dir = goal_handle.request.grasp_direction
        orientation = goal_handle.request.orientation
        is_pick = goal_handle.request.pick

        if not is_pick:
            return self._abort(goal_handle, 'This action only supports pick=True')

        x, y, z = float(rough_pos[0]), float(rough_pos[1]), float(rough_pos[2])

        cam_frame = self.get_parameter('gripper_camera_frame').value

        self.get_logger().info(
            f'[gripper_pick] Start "{obj_name}" at map({x:.3f},{y:.3f},{z:.3f}) '
            f'grasp={grasp_dir}'
        )

        # Ensure intrinsics are available
        if not self._intrinsics_ready:
            for p, name in [(self.get_parameter('fx').value, 'fx'),
                            (self.get_parameter('fy').value, 'fy'),
                            (self.get_parameter('cx').value, 'cx'),
                            (self.get_parameter('cy').value, 'cy')]:
                if p <= 0.0:
                    return self._abort(goal_handle, f'Intrinsics not ready and param {name} invalid')
            self._fx, self._fy = self.get_parameter('fx').value, self.get_parameter('fy').value
            self._cx, self._cy = self.get_parameter('cx').value, self.get_parameter('cy').value
            self._intrinsics_ready = True

        # ── Stage 1: move arm to SCAN pose ──────────────────────────────
        self._publish_feedback(goal_handle, 'moving_to_scan', 0.1)
        try:
            scan_pose = self._compute_scan_pose(x, y, z, grasp_dir, orientation)
        except Exception as e:
            return self._abort(goal_handle, f'Scan pose compute failed: {e}')

        ok, msg = await self._move_arm_to_pose(scan_pose)
        if not ok:
            return self._abort(goal_handle, f'Move to scan pose failed: {msg}')

        time.sleep(0.5)  # let vibration settle

        # ── Stage 2: depth-based refinement ────────────────────────────────
        self._publish_feedback(goal_handle, 'depth_refinement', 0.3)

        # Expected position in camera frame (from rough map position)
        try:
            expected_cam = self._transform_point(
                x, y, z, 'map', cam_frame
            )
        except Exception as e:
            return self._abort(goal_handle, f'TF map->camera failed: {e}')

        refined_cam = self._refine_from_depth(expected_cam)
        if refined_cam is None:
            return self._abort(goal_handle, 'Depth refinement failed: no cluster found')

        # Transform refined point back to map
        try:
            refined_map = self._transform_point(
                refined_cam[0], refined_cam[1], refined_cam[2],
                cam_frame, 'map'
            )
        except Exception as e:
            return self._abort(goal_handle, f'TF camera->map failed: {e}')

        self.get_logger().info(
            f'[gripper_pick] Refined: map({refined_map[0]:.3f}, '
            f'{refined_map[1]:.3f}, {refined_map[2]:.3f})'
        )

        # ── Stage 3: call existing arm manipulation with refined pos ─────
        self._publish_feedback(goal_handle, 'executing_pick', 0.7)

        arm_goal = ArmManipulation.Goal()
        arm_goal.object_name = obj_name
        arm_goal.object_position = [float(v) for v in refined_map]
        arm_goal.grasp_direction = grasp_dir
        arm_goal.pick = True

        self._arm_client.wait_for_server()
        send_goal = await self._arm_client.send_goal_async(arm_goal)
        if not send_goal.accepted:
            return self._abort(goal_handle, 'Arm manipulation goal rejected')

        result_resp = await send_goal.get_result_async()
        if result_resp.result.success:
            self._publish_feedback(goal_handle, 'done', 1.0)
            goal_handle.succeed()
            res = ArmManipulation.Result()
            res.success = True
            res.message = f'Picked {obj_name} with gripper depth refinement'
            return res
        else:
            return self._abort(goal_handle, f'Arm manipulation failed: {result_resp.result.message}')

    # ------------------------------------------------------------------
    # Depth-only refinement (the core "no color" logic)
    # ------------------------------------------------------------------
    def _refine_from_depth(self, expected_cam: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert latest depth image to a point cloud in camera frame,
        keep points inside a sphere around `expected_cam`, and return
        the median of the cluster.
        """
        with self._depth_lock:
            depth = self._latest_depth.copy() if self._latest_depth is not None else None

        if depth is None:
            self.get_logger().warn('No depth frame received yet')
            return None

        fx, fy, cx, cy = self._fx, self._fy, self._cx, self._cy
        h, w = depth.shape

        # Build image coordinate grids
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)

        z = depth
        valid = (z > 0.05) & (z < self.get_parameter('depth_max').value) & np.isfinite(z)

        # Back-project valid pixels to 3-D in camera optical frame
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy

        points = np.stack([x, y, z], axis=-1)          # H x W x 3
        points = points[valid]                          # N x 3

        min_pts = self.get_parameter('min_cluster_points').value
        if points.shape[0] < min_pts:
            self.get_logger().warn(f'Too few valid depth points: {points.shape[0]}')
            return None

        # Keep points near the expected position (geometric search radius)
        search_r = self.get_parameter('search_radius').value
        dists = np.linalg.norm(points - expected_cam, axis=1)
        cluster = points[dists < search_r]

        if cluster.shape[0] < min_pts:
            self.get_logger().warn(f'Cluster too small: {cluster.shape[0]} points')
            return None

        refined = np.median(cluster, axis=0)
        self.get_logger().info(
            f'[depth] expected={expected_cam}  refined={refined}  '
            f'pts={cluster.shape[0]}'
        )
        return refined

    # ------------------------------------------------------------------
    # MoveIt: move arm to a pose (used for the SCAN pose)
    # ------------------------------------------------------------------
    async def _move_arm_to_pose(self, target_pose: Pose):
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'arm_group'
        req.pipeline_id = 'ompl'          # same pipeline RViz uses
        req.planner_id = ''               # let OMPL pick the default

        # ── Start state: current robot state ─────────────────────────────
        req.start_state.is_diff = False
        with self._joint_state_lock:
            if self._latest_joint_state is not None:
                req.start_state.joint_state = self._latest_joint_state
            else:
                self.get_logger().warn('No /joint_states received yet; using default start state')

        req.num_planning_attempts = 20
        req.allowed_planning_time = 15.0
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        # ── Position constraint (slightly relaxed sphere) ────────────────
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = 'g_base'
        pos_constraint.link_name = 'tool_tip'
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.01]     # 1 cm tolerance (was 0.001)
        bv = BoundingVolume()
        bv.primitives.append(primitive)
        bv.primitive_poses.append(target_pose)
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0

        # ── Orientation constraint (tight for actual pose) ───────────────
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
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 5
        goal.planning_options.replan_delay = 1.0

        self.get_logger().info(
            f'[MoveIt] Planning to g_base({target_pose.position.x:.3f}, '
            f'{target_pose.position.y:.3f}, {target_pose.position.z:.3f})'
        )

        self._move_client.wait_for_server()
        send_goal = await self._move_client.send_goal_async(goal)
        if not send_goal.accepted:
            return False, 'MoveIt goal rejected'

        result = await send_goal.get_result_async()
        if result.result.error_code.val == 1:
            return True, 'ok'
        return False, f'MoveIt error_code={result.result.error_code.val}'

    # ------------------------------------------------------------------
    # TF helper: transform a 3-D point between frames
    # ------------------------------------------------------------------
    def _transform_point(self, x, y, z, from_frame, to_frame):
        p = PointStamped()
        p.header.frame_id = from_frame
        p.point.x = float(x)
        p.point.y = float(y)
        p.point.z = float(z)

        transform = self._tf_buffer.lookup_transform(
            to_frame, from_frame, rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=3.0),
        )
        p_out = tf2_geometry_msgs.do_transform_point(p, transform)
        return np.array([p_out.point.x, p_out.point.y, p_out.point.z])

    # ------------------------------------------------------------------
    # Compute a SCAN pose in g_base offset from the object
    # ------------------------------------------------------------------
    def _compute_scan_pose(self, obj_x, obj_y, obj_z, grasp_dir, orientation):
        """
        Place the gripper (and therefore the camera) back from the object
        along the approach direction so the object is comfortably in view.
        """
        scan_dist = self.get_parameter('scan_distance').value

        # Object position in g_base
        p_base = self._transform_point(obj_x, obj_y, obj_z, 'map', 'g_base')

        pose = Pose()
        pose.position.x = p_base[0]
        pose.position.y = p_base[1]
        pose.position.z = p_base[2]

        if grasp_dir == 'Top':
            rx, ry, rz = GRASP_ORIENTATIONS['Top']
            pose.position.z += scan_dist          # camera above object
        elif grasp_dir == 'Front':
            rx, ry, rz = GRASP_ORIENTATIONS['Front']
            pose.position.x -= scan_dist          # camera back from object
        else:
            rx, ry, rz = GRASP_ORIENTATIONS['Front']
            pose.position.x -= scan_dist

        # rx, ry, rz = float(orientation[0]), float(orientation[1]), float(orientation[2])
        

        r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        qx, qy, qz, qw = r.as_quat()
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    # ------------------------------------------------------------------
    # Feedback / abort helpers
    # ------------------------------------------------------------------
    def _publish_feedback(self, goal_handle: ServerGoalHandle, stage: str, progress: float):
        fb = ArmManipulation.Feedback()
        fb.stage = stage
        fb.progress = progress
        goal_handle.publish_feedback(fb)
        self.get_logger().info(f'[feedback] {stage} ({progress:.0%})')

    def _abort(self, goal_handle: ServerGoalHandle, message: str):
        self.get_logger().error(f'[gripper_pick] ABORT: {message}')
        res = ArmManipulation.Result()
        res.success = False
        res.message = message
        goal_handle.abort()
        return res


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = GripperDepthPicker()
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