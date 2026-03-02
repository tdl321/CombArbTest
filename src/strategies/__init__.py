"""Strategy layer for the modular pipeline.

Strategies are self-registering plugins. Import a strategy module
to register it with the strategy registry.
"""

# Import strategies to trigger registration
from . import combinatorial_arb  # noqa: F401
