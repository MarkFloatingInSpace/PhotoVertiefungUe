# -*- coding: cp1252 -*-

import _prolog
from oriental import stereo
from oriental.utils.argparse import exitCode

__doc__ = stereo.__doc__

if __name__ == '__main__':
    exitCode( stereo.parseArgs )