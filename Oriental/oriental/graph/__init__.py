# -*- coding: cp1252 -*-
"""
oriental.graph - graphs for oriental
====================================
"""

__author__   = """VIAS - Vienna Institute for Archaeological Science
Interdisziplinäre Forschungsplattform Archäologie
Universität Wien
Franz-Klein-Gasse 1/III
1190 Wien
Austria"""

__license__  = "?"
__version__  = "0.0.1"
__date__     = "2013"
__url__      = "http://vias.univie.ac.at/"

from .. import config
if config.debug:
    from ._graph_d import *
    from ._graph_d import __date__
else:
    from ._graph import *
    from ._graph import __date__
    
        
from ..utils.strProperties import strProperties

# If nested classes (or enum's nested in a class) are exported to Python using boost::python::scope,
# then pickling of those nested classes is impossible.
# That is not a problem of boost::python, but of Python's pickle in general (the problem also exists for nested classes defined in Python)!
# Thus, let's export all nested classes to Python in the module scope.
# On the Python-side (graph/__init__.py), we can still attach the nested classes as attributes to their surrounding classes, see below.
# Note: pickling of boost::python::enum_'s only works with the highest pickling protocol.
#ImageConnectivity.Image = ImageConnectivity_Image
#ImageConnectivity.Image.State = ImageConnectivity_Image_State
#ImageConnectivity.Edge = ImageConnectivity_Edge
#ImageConnectivity.Edge.State = ImageConnectivity_Edge_State

ImageFeatureID.__str__  = strProperties
ImageFeatureID.__repr__ = strProperties

ImageConnectivity.Image.__str__  = strProperties
ImageConnectivity.Image.__repr__ = strProperties

ImageConnectivity.Edge.__str__  = strProperties
ImageConnectivity.Edge.__repr__ = strProperties

# Pickle supports boost::python::enum_ when using the highest pickle protocol.
# Pickle support for boost::python::enum_ nested within a class (that can be a nested class, again)
#   can be introduced here, on the Python-side, by converting them forth and back to int
# The following solution was taken from http://stackoverflow.com/questions/3214969/pickling-an-enum-exposed-by-boost-python
#   adapted to allow for enum's defined not only in module scope, but also within classes.

# The following could probably be extended to support pickling of nested classes, too, see http://stackoverflow.com/questions/1947904/how-can-i-pickle-a-nested-class-in-python
# However, it seems easier to instead export all classes at module scope, and on import in Python,
# assign the nested classes as attributes to their surrounding class.
##def _isEnumType(o):
##    return isinstance(o, type) and issubclass(o,int) and not (o is int)
##
##def _tuple2enum(scope, value):
##    enum = None
##    for idx in range(len(scope)):
##        if idx==0:
##            enum = getattr(_myCExt, scope[idx])
##        else:
##            enum = getattr(enum, scope[idx])
##
##    e = enum.values.get(value,None)
##    if e is None:
##        e = enum(value)
##    return e
##
##def _registerEnumPicklers(): 
##    from copy_reg import constructor, pickle
##    from functools import partial as _partial
##    def reduce_enum(e, scope=[]):
##        enum = type(e).__name__.split('.')[-1]
##        scope2 = scope[:]
##        scope2.append( enum )
##        return ( _tuple2enum, ( scope2, int(e) ) )
##    constructor( _tuple2enum)
##    def reduceRecursively( d, scope = [] ):
##        for item in d.values():
##            if _isEnumType(item):
##                pickle(item, _partial( reduce_enum, scope=scope ) )
##            if hasattr( item, '__dict__' ):
##                scope2 = scope[:]
##                scope2.append( item.__name__ )
##                reduceRecursively( item.__dict__, scope2 )
##     
##    reduceRecursively( _myCExt.__dict__ )
##           
##_registerEnumPicklers()