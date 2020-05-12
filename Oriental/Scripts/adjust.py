# -*- coding: cp1252 -*-

import _prolog
from oriental import adjustScript
from oriental.utils.argparse import exitCode

__doc__ = adjustScript.__doc__

if __name__ == '__main__':
    exitCode( adjustScript.parseArgs )