"""Perception and VLM interfaces.

Two backends are provided:

* ``SimPerception`` / ``SimVLM``: deterministic, offline, reads simulator ground
  truth (optionally with injected noise/dropout).  This is what tests and the
  50x3x3 evaluation use, so numbers are reproducible on a 2-core CPU.
* ``OpenAICompatVLM``: a real HTTP adapter for a Qwen2.5-VL style OpenAI-compatible
  endpoint.  It is **not** exercised by the test suite (no credentials, no GPU on
  this offline workstation) and is documented as such in the README.
"""

from agentos.perception.base import PerceptionBackend, VLMBackend
from agentos.perception.openai_vlm import OpenAICompatVLM
from agentos.perception.sim_perception import SimPerception
from agentos.perception.sim_vlm import SimVLM

__all__ = ["PerceptionBackend", "VLMBackend", "SimPerception", "SimVLM", "OpenAICompatVLM"]
