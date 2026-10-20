"""Pytest bootstrap.

The project keeps pytest's ``--basetemp`` inside ``.cache/pytest`` so that no
temporary artifacts leak into the repository root.  A fresh checkout does not
contain ``.cache`` yet (it is git-ignored), and pytest does not create missing
parent directories for ``--basetemp``; create it before the temp path factory is
used.  This root conftest is imported before any test module, which makes the
same command work on a clean clone in CI and on a developer machine.
"""

from pathlib import Path

Path(__file__).resolve().parent.joinpath(".cache").mkdir(exist_ok=True)
