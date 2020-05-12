# -*- coding: cp1252 -*-
"""create a human readable string of the properties and resp. values of a class instance, excluding functions, etc."""
def strProperties( self ):
    namesValues = list()
    longestName = 0
    for attr, value in self.__class__.__dict__.items():
        if isinstance( value, property ):
            #print(attr)
            longestName = max( len(attr), longestName )
            try:
                namesValues.append( ( attr, getattr( self, attr ) ) )
            except Exception:
                # _oriental_d.Exception: Memory of this pointer of type oriental::adjust_py::ParameterBlockOrdering has already been released!
                namesValues.append( ( attr, "<failed to extract value, type:{}>".format( value.__class__.__name__ ) ) )
                
    
    # The class dict is unordered, like any other dict, so the original order of the definition of attributes is not accessible here.
    # Thus, at least sort the names lexicographically. 
    namesValues.sort()
    if 0:
        # formatting on multiple lines results in problems e.g. with sphinx, because sphinx interprets the string as restructured text, where indenting has special meanings
        # e.g. for adjust.Problem.Evaluate (which has an argument of type Problem::EvaluateOptions, which is rendered with a line-break by strProperties):
        # docstring of oriental.adjust._adjust.Problem.Evaluate:3: WARNING: Block quote ends without a blank line; unexpected unindent.
        fmtString = "{:>" + str(longestName)  + "} : {}\n"
        els = ( fmtString.format( *nameValue ) for nameValue in namesValues )
        return ''.join(els)

    els = ( "{}={}".format( *nameValue ) for nameValue in namesValues )
    return self.__class__.__name__ + "(" + ', '.join(els) + ")"
