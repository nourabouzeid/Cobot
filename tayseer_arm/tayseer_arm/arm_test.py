#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from tayseer_interfaces.action import ArmManipulation


class TestArmManipulationClient(Node):
    def __init__(self):
        super().__init__(
            'test_arm_manipulation_client',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.parameter.Parameter.Type.BOOL,
                    True,
                )
            ],
        )
        self._client = ActionClient(self, ArmManipulation, '/arm_manipulate')

    def send_goal(self, object_name, position, grasp_direction, pick: bool):
        self._client.wait_for_server()

        mode = 'pick' if pick else 'place'
        self.get_logger().info(f"Server found, sending {mode} goal...")

        goal = ArmManipulation.Goal()
        goal.object_name      = object_name
        goal.object_position  = [float(p) for p in position]
        goal.grasp_direction  = grasp_direction
        goal.pick             = pick

        future = self._client.send_goal_async(
            goal,
            feedback_callback=self._feedback_cb,
        )
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'{mode.capitalize()} goal rejected!')
            return False

        self.get_logger().info(f'{mode.capitalize()} goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.success:
            self.get_logger().info(f'SUCCESS: {result.message}')
            return True
        else:
            self.get_logger().error(f'FAILED: {result.message}')
            return False

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f'[{fb.stage}] {fb.progress:.0%}')


def main(args=None):
    rclpy.init(args=args)
    node = TestArmManipulationClient()

    pick_position  = [0.26343, -0.01358, 0.09]
    place_position = [0.280000,  -0.1000, 0.15]
    object_name    = 'fire_extinguisher'
    grasp_direction = 'Top'   # 'Top' or 'Front'

    # --- Step 1: Pick ---
    success = node.send_goal(
        object_name     = object_name,
        position        = pick_position,
        grasp_direction = grasp_direction,
        pick            = True,
    )

    # --- Step 2: Place (only if pick succeeded) ---
    if success:
        node.send_goal(
            object_name     = object_name,
            position        = place_position,
            grasp_direction = "Front",
            pick            = False,
        )
    else:
        node.get_logger().error('Pick failed — skipping place.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()