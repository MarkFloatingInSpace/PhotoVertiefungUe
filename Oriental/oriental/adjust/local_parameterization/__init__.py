# -*- coding: cp1252 -*-
"""
provides local parameterizations to be used by oriental.adjust.
"""

from ... import config as _config, _setup_summary_docstring

if _config.debug:
    from ._local_parameterization_d import *
    from ._local_parameterization_d import __date__
    thePyd = _local_parameterization_d
else:
    from ._local_parameterization import *
    from ._local_parameterization import __date__
    thePyd = _local_parameterization

# The _AutoDiff_X_Y classes are not imported above, as their names begin with an underscore. We name them that way to exclude them from the inheritance graph in the docs.
for _name in dir( thePyd ):
    if _name.startswith('_AutoDiff'):
        globals()[_name] = getattr( thePyd, _name )
del thePyd

from ...utils.strProperties import strProperties as _strProperties

# enable print(Options)
# Much more easily implemented in Python than in C++!
for _typ in ( Identity, Subset, Quaternion, UnitSphere, Sphere ):
    _typ.__str__  = _strProperties
    _typ.__repr__ = _typ.__str__

def _summary():
    pass

_setup_summary_docstring( _summary, __name__ )