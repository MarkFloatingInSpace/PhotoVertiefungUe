# -*- coding: cp1252 -*-

import _prolog
from oriental import import_
from oriental.utils.argparse import exitCode

__doc__ = import_.__doc__

if __name__ == '__main__':
    exitCode( import_.parseArgs )