"""Compatibility alias; canonical implementation is in spy_var.models.classical."""
import sys
from .models import classical as _implementation
sys.modules[__name__] = _implementation
