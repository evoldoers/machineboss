"""machineboss - Python interface for Machine Boss WFST toolkit.

Core types:
    Machine, MachineState, MachineTransition  - JSON WFST format
    EvaluatedMachine                          - Tokenized + numerically evaluated
    TransMachine                              - JAX pytree (primary JAX interface)
"""

__version__ = "0.1.0"

from .machine import Machine, MachineState, MachineTransition
from .eval import EvaluatedMachine
