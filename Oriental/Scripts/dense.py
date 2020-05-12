# -*- coding: cp1252 -*-

import _prolog
from oriental import dense
from oriental.utils.argparse import exitCode

__doc__ = dense.__doc__

if __name__ == '__main__':
    exitCode( dense.parseArgs )