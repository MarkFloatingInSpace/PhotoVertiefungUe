# -*- coding: cp1252 -*-
"""Hierarchical least squares multi image matching"""

from ... import config as _config

if _config.debug:
    from ._lsm_d import *
    from ._lsm_d import __date__
else:
    from ._lsm import *
    from ._lsm import __date__

from ..strProperties import strProperties

for typ in ( TrafoAffine2D, Image, IterationSummary, ResolutionLevelSummary, SolveSummary, SolveOptions, Lsm ):
    typ.__str__  = strProperties
    typ.__repr__ = typ.__str__