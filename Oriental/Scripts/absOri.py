# -*- coding: cp1252 -*-

import _prolog
from oriental.absOri import main
from oriental.utils.argparse import exitCode

__doc__ = main.__doc__

if __name__ == '__main__':
    exitCode( main.parseArgs )