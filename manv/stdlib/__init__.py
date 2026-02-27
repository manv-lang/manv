"""ManV standard library v1 surface.

This package provides a broad, Python-comparable stdlib surface for ManV
runtime integration. Logical std module names are mapped through
``module_index`` so the language/runtime can evolve independently of Python
package naming constraints.
"""

from .module_index import LOGICAL_MODULES, ModuleSpec, load_std_module

__all__ = [
    "LOGICAL_MODULES",
    "ModuleSpec",
    "load_std_module",
]
