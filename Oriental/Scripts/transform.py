# -*- coding: cp1252 -*-

import _prolog
from oriental import transform
from oriental.utils.argparse import exitCode

__doc__ = transform.__doc__

if __name__ == '__main__':
    exitCode( transform.parseArgs )