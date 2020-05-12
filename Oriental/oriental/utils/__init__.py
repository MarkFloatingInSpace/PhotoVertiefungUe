# -*- coding: cp1252 -*-
"""anything that does not justify an own package"""

from .. import Progress

import itertools
from contextlib import contextmanager
import inspect
import pkgutil
import sys
import os.path

from contracts import contract

def importSubPackagesAndModules( packagePath = None, packageName = None ):
    """call this in a function defined in __init__.py of a package, and it loads all direct sub-packages into the namespace of the package. 
       usage within an __init__.py::

          def importSubPackagesAndModules():
              utils.importSubPackagesAndModules( __path__, __name__, globals() )
    """
    packagePath = packagePath or [os.path.dirname(__file__)]
    packageName = packageName or 'oriental.utils'
    theMod = sys.modules[packageName]
    for importer, modname, ispkg  in pkgutil.iter_modules(packagePath):
        #if not ispkg:
        #    continue
        if modname.startswith('_'):
            # Don't import private modules, including release/debug-versions of our C++ extension modules. 
            # It is impossible to load debug-versions in a release environment and vice versa.
            continue
        fullName = packageName + '.' + modname
        if fullName in sys.modules:
            continue
        module = importer.find_module(fullName).load_module(fullName)
        theMod.__dict__[modname] = module

def formatBytes(nBytes,precision=0):
    fmt = '{{:.{}f}}{{}}'.format(precision)
    for name in ('B','KB','MB','GB','TB'):
        if nBytes < 1024:
            return fmt.format(nBytes,name)
        nBytes /= 1024
    return fmt.format(nBytes,'PB')

@contextmanager
@contract
def progressAfter(progress : Progress) -> None:
    yield
    progress += 1


class IterableLengthMismatch(ValueError):
    pass


class CallFnCounted:
    # cannot use contracts here, as it does not support kwonlyargs
    def __init__( self, func, *args, nDigits : int = 2, **kwargs ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.counter = 0
        self.nDigits = nDigits
        self.lastFn = None
        # determine the index of argument 'fn' of func
        sig = inspect.signature(func)
        for idx,param in enumerate(sig.parameters):
            if param == 'fn':
                self.fnPos = idx
                break
        else:
            raise Exception("func must have 'fn' as argument")

    def __call__( self, *args, **kwargs ):
        self.counter += 1
        kwargsMerged = self.kwargs.copy()
        kwargsMerged.update(kwargs) # overwrite existing entries
        try:
            fn =  kwargsMerged.pop('fn')
        except KeyError:
            raise Exception("'fn' must be passed as keyword argument")
        fn = fn.parent / '{:0{}}_{}'.format( self.counter, self.nDigits, fn.name )
        self.lastFn = fn
        argsMerged = self.args + args
        if len(argsMerged) > self.fnPos:
            argsMerged = argsMerged[:self.fnPos] + (fn,) + argsMerged[self.fnPos:]
            return self.func( *argsMerged, **kwargsMerged )
        return self.func( *argsMerged, fn=fn, **kwargsMerged )

#def zip_equal(*iterables):
#    """equivalent to built-in zip, but raises if iterables have unequal lengths
#    The only situations in which the behaviour of built-in zip may be wanted
#    is when zipping infinite iterators (e.g. itertools.repeat(.)) and iterables of finite length!
#    Using yield from (new in Python 3.3), surely most efficient version.
#    However, it doesn't work if the first iterable is longer than the second,
#    because zip() advances the first iterator, then fails to advance the second, and returns (raises StopIteration)."""
#    iterators = tuple(iter(el) for el in iterables)
#    yield from zip(*iterators) # returns None, so the return value cannot be used to determine if any of the iterators have already been advanced but not yielded.
#    # shortest iterable is done, see if there are any values left
#    sentinel = object()
#    if any([next(it, sentinel) is not sentinel for it in iterators]):
#        raise IterableLengthMismatch

def zip_equal(*iterables):
    """equivalent to built-in zip, but raises if iterables have unequal lengths
    The only situations in which the behaviour of built-in zip may be wanted
    is when zipping infinite iterators (e.g. itertools.repeat(.)) and iterables of finite length!"""
    sentinel = object()
    for els in itertools.zip_longest( *iterables, fillvalue=sentinel ):
        #if sentinel in els: # 'in' seems to compare by-value. Doesn't work e.g. with numpy.ndarray
        if any( el is sentinel for el in els ):
            raise IterableLengthMismatch
        yield els

def iround(number):
    "round to int, even for numpy.int32, np.float64, etc."
    return int(round(number))

#def zip_equal(*iterables):
#    'works, but is a bit lengthy'
#    iterators = [iter(x) for x in iterables]
#    while True:
#        try:
#            first_value = next(iterators[0])
#            try:
#                # we cannot create a tuple from a generator expression instead,
#                # because currently, generator epressions never raise StopIteration: http://stackoverflow.com/questions/16814111/generator-expression-never-raises-stopiteration
#                other_values = [next(x) for x in iterators[1:]]
#            except StopIteration:
#                raise IterableLengthMismatch
#            else:
#                values = [first_value] + other_values
#                yield tuple(values)
#        except StopIteration:
#            for iterator in iterators[1:]:
#                try:
#                    extra_value = next(iterator)
#                except StopIteration:
#                    pass # this is what we expect
#                else:
#                    raise IterableLengthMismatch
#            raise StopIteration


# cannot import utils.strProperties into its own init script :-(

# legacy code, used before our own Python binding for gdal had been established
#from contracts import contract
#import numpy as np
#from osgeo import gdal
#
#gdal2np = {
#    gdal.GDT_Byte : np.uint8,
#    gdal.GDT_UInt16 : np.uint16,
#    gdal.GDT_Int16 : np.int16,
#    gdal.GDT_UInt32 : np.uint32,	
#    gdal.GDT_Int32 : np.int32,
#    gdal.GDT_Float32 : np.float32,
#    gdal.GDT_Float64 : np.float64,
#    #gdal.GDT_CInt16 	
#    #gdal.GDT_CInt32 	
#    #gdal.GDT_CFloat32 	
#    gdal.GDT_CFloat64 : np.complex64
#}
#
#@contract
#def imreadBGR( fn : str,
#               width_px : 'int|None' = None
#             ) -> 'array[NxMx3]':
#    "openCV reads images as BGR. For plotting with openCV, use img as is."
#
#    ds = gdal.Open( fn, gdal.GA_ReadOnly )
#    height_px = round( width_px * ds.RasterYSize / ds.RasterXSize ) if width_px is not None else None
#    gdalDt = ds.GetRasterBand(1).DataType
#    gdalDtSz = gdal.GetDataTypeSize( gdalDt ) // 8 # GetDataTypeSize returns #bits!
#    buf_xsize = ds.RasterXSize if width_px  is None else width_px
#    buf_ysize = ds.RasterYSize if height_px is None else height_px
#    buffer = ds.ReadRaster( xoff=0, yoff=0,
#                            xsize=ds.RasterXSize, ysize=ds.RasterYSize,
#                            buf_xsize=buf_xsize, buf_ysize=buf_ysize,
#                            buf_pixel_space=gdalDtSz*ds.RasterCount, buf_line_space=gdalDtSz*ds.RasterCount*buf_xsize, buf_band_space=gdalDtSz )
#    img = np.ndarray( (buf_ysize,buf_xsize,ds.RasterCount), dtype=gdal2np[gdalDt], buffer=buffer ).squeeze()
#    colorInterpretations = [ ds.GetRasterBand(idx+1).GetColorInterpretation() for idx in range(ds.RasterCount) ]
#    if colorInterpretations == [ gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand ]:
#        img = img[:,:,::-1]
#    elif colorInterpretations == [ gdal.GCI_BlueBand, gdal.GCI_GreenBand, gdal.GCI_RedBand ]:
#        pass
#    elif colorInterpretations == [ gdal.GCI_GrayIndex ]:
#        img = np.dstack((img,img,img))
#    else:
#        # For palette files, we might fallback to reading in the data using ReadAsArray(), and then downsampling
#        raise Exception( "(Combination of) channels unsupported: {}".format( ','.join(( gdal.GetColorInterpretationName(col) for col in colorInterpretations )) ) )
#    return img
#
#@contract
#def imreadRGB( fn : str,
#               width_px : 'int|None' = None
#             ) -> 'array[NxMx3]':
#    "For plotting with matplotlib, convert to RGB"
#    return imreadBGR( fn, width_px )[:,:,::-1]
