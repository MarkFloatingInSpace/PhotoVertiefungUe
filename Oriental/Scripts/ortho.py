# -*- coding: cp1252 -*-

import _prolog
from oriental import ortho
from oriental.utils.argparse import exitCode

__doc__ = ortho.__doc__

if __name__ == '__main__':
    exitCode( ortho.parseArgs )