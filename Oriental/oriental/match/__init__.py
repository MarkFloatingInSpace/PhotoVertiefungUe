# -*- coding: cp1252 -*-
"""
image feature extraction and matching
"""

from .. import config
if config.debug:
    from ._match_d import *
    from ._match_d import __date__
else:
    from ._match import *
    from ._match import __date__
    
from ..utils.strProperties import strProperties

EdgeMatch.__str__ = strProperties
EdgeMatch.__repr__ = EdgeMatch.__str__
SurfOptions.__repr__ = SurfOptions.__str__
SiftOptions.__repr__ = SiftOptions.__str__
AkazeOptions.__repr__ = AkazeOptions.__str__
FeatureFiltOpts.__str__ = strProperties
FeatureFiltOpts.__repr__ = FeatureFiltOpts.__str__
MatchingOpts.__str__ = strProperties
MatchingOpts.__repr__ = MatchingOpts.__str__
