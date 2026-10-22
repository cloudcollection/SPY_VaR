"""Compatibility alias; canonical implementation is in spy_var.models.garch_anchored."""
import sys
from .models import garch_anchored as _implementation
sys.modules[__name__] = _implementation
