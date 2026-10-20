"""Minimal fake manipulator node for Gazebo smoke tests.

It logs the received skill and reports success so the whole AgentOS loop can be
exercised on a Linux host without a real arm.  Never used in evaluation.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node


class FakeManipulator(Node):
    def __init__(self) -> None:
        super().__init__("agentos_fake_manipulator")
        self.declare_parameter("log_level", "info")
        self.get_logger().info("fake manipulator ready (no hardware): pick/place return success")

    def fake_execute(self, skill: str, params: dict) -> bool:
        self.get_logger().info(f"fake manipulator executed {skill} with {params}")
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeManipulator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
