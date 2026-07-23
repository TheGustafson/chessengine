"""ARCHON NNUE training in JAX -- D47c-parity prototype (2026-07-22).

CPU-only by policy while the bullet trainer owns the GPU: every entry point
must run with ``JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false``.
"""

from . import spec  # noqa: F401
