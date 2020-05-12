# -*- coding: cp1252 -*-
"""
loss functors for oriental.adjust
"""

from ... import config as _config, _setup_summary_docstring

if _config.debug:
    from ._loss_d import *
    from ._loss_d import __date__
else:
    from ._loss import *
    from ._loss import __date__

def _summary():
    pass

_setup_summary_docstring( _summary, __name__ )