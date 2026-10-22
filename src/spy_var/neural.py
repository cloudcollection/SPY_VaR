"""Compatibility alias; canonical implementation is in spy_var.models.mlp."""
import sys
from .models import mlp as _implementation
sys.modules[__name__] = _implementation
