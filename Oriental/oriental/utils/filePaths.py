from contracts import contract, new_contract

import os
from pathlib import Path

new_contract('Path',Path)

class ShortFileNames():

    @contract
    def __init__( self, fns : 'list(str|Path)' ):
        # index range to select the part of file names that differ (excluding the common prefix and common suffix) as shortened image names 
        # Note that -0 == 0 ! If the files don't share a common file extension, then we need to extend the slice until the end of every string, where the strings may have different lengths. This is done with text[start:None]; Thus: ... or None
        self.substringRange = (  len(os.path.commonprefix([ str(fn).lower()       for fn in fns])),
                                -len(os.path.commonprefix([ str(fn)[::-1].lower() for fn in fns])) or None )

        self.question = "?" * len(str(fns[0])[self.substringRange[0] : self.substringRange[1]])
        
        # arbitrarily use the one of the first pho
        self.commonName = str(fns[0])[ 0 : self.substringRange[0]] + self.question + ( str(fns[0])[self.substringRange[1] : ] if self.substringRange[1] is not None else '' )

    @contract
    def __call__( self, fn : 'str|Path' ):
        return str(fn)[ self.substringRange[0] : self.substringRange[1] ]

def relPathIfExists( path, start ):
    try:
        return os.path.relpath(str(path), str(start))
    except ValueError:
        return path