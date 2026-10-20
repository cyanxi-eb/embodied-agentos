"""Embodied AgentOS: a small but real robot-agent runtime.

Layers (bottom-up):

* ``agentos.sim``      deterministic 2D warehouse simulator + ROS2-compatible skill backend
* ``agentos.skills``   8 robot skills with JSON-schema params, pre/post conditions, failure codes
* ``agentos.safety``   unified safety gateway (speed clamp, latched e-stop, timeout, geofence, deny-list)
* ``agentos.bt``       py_trees behavior tree: execute -> verify -> retry / backtrack / request replan
* ``agentos.graph``    LangGraph Plan-and-Act state machine (checkpointed, interruptible)
* ``agentos.memory``   Qdrant scene/task memory + SQLite session & trace store
* ``agentos.api``      FastAPI + SSE debug console
* ``agentos.mcp_server`` MCP stdio server exposing the skill library
* ``agentos.ros2_bridge`` rclpy adapter (import-guarded; not run on the development host)
"""

__version__ = "0.1.0"
__author__ = "陈彦希"
