# -*- coding: cp1252 -*-
##import collections
##
### using types.MappingProxyType instead of dict for Parameter.parpar
### would prevent users from accidentally destroying the numpy.ndarray that adjust holds a double* or int* pointer of
### by re-assigning Parameter.parpar["key"] to another ndarray <- seems like an important benefit!
### But types.MappingProxyType does not derive from dict. How to access it in C++?
    
# _FrozenDict derives from dict, and thus it's easy to access its values in C++
# prevents the following:
# omfika1.parpar["eulerAngles"] = adjust.EulerAngles.alzeka
# omfika1.parpar["eulerAngles"] = 5
# del omfika1.parpar["eulerAngles"]
# However, the following is still okay, because ndarray is writable:
# omfika1.parpar["eulerAngles"][0] = adjust.EulerAngles.alzeka

class _FrozenDict(dict):
    def __setitem__(self, key, val):
        if key in self:
            raise Exception("Must not re-assign values. Mutate them instead")
        super().__setitem__(key, val)
    
    def __delitem__(self,key):
        raise Exception("Must not delete entries.")

from contracts import contract, new_contract
import numpy as np
import inspect
from .. import adjust

# http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
# see http://www.scipy.org/Subclasses

class Array(np.ndarray):
    """Sub-class of :class:`numpy.ndarray` that can be constructed using the signature of :func:`numpy.array`"""
    __slots__ = ()
    def __new__(subtype, array = None ):
        if array is not None:
            obj = np.asarray(array,float).reshape((len(subtype.staticNames),))
        else:
            obj = np.zeros(len(subtype.staticNames))
        return obj.view(subtype)

    @property
    def names(self) -> str:
        return self.staticNames


class ParPar( Array ):
    """A :class:`numpy.ndarray` with the additional dict-attribute :attr:`parpar`.

    :attr:`parpar` itself holds :class:`numpy.ndarray`\s that provide C-pointers to be used in C++. 
    Don't use directly, but only derived classes."""
    __slots__ = 'parpar'
    def __new__(subtype, array, parpar=None):

        # Make sure we are working with an array, and copy the array if requested
        obj = super().__new__(subtype,array)

        # Transform 'subarr' from an ndarray to our new subclass.
        #subarr = subarr.view(subtype)

        # Use the specified 'parpar' parameter if given
        if parpar is None:
            parpar = getattr(array, 'parpar', None)
        if parpar is not None:
            obj.parpar = parpar

        if not isinstance(obj.parpar,_FrozenDict):
            raise Exception("parpar must be of type _FrozenDict")

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self,obj):
        # We use the getattr method to set a default if 'obj' doesn't have the 'parpar' attribute
        self.parpar = getattr(obj, 'parpar', _FrozenDict())

    # support pickle
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.parpar,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    # support pickle
    def __setstate__(self, state):
        self.parpar = state[-1]  # Set the parpar attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[:-1])

    def _getParParsFormatted(self):
        for key in self.parpar:
            name = key.split('.')[-1]
            yield '{}={}'.format(name,getattr(self,name))

    def __repr__(self):
        return "{} {}".format( np.ndarray.__repr__(self), ' '.join( self._getParParsFormatted()) )

    def __str__(self):
        return "{} {}".format( np.ndarray.__str__(self), ' '.join( self._getParParsFormatted()) )

# The following would be using multiple inheritance. Works, but is very slow!
#class Info:
#    """Sub-class of numpy.ndarray with additional attribute 'info'
#    that can be constructed using the signature of numpy.array"""
#    __slots__ = ()
#    @contract
#    def __new__(cls,
#                info=None,
#                *args, **kwargs
#               ) -> 'array[N](float)':
#        obj = super().__new__(cls,*args, **kwargs)
#        # add the new attribute to the created instance
#        #if info is None:
#        #    info = getattr(obj,'info',None)
#        if info is not None:
#            obj.info = info
#        # Finally, we must return the newly created object:
#        return obj
#
#    def __array_finalize__(self, obj):
#        superFinalize = super().__array_finalize__
#        if superFinalize is not None:
#            superFinalize(obj)
#        if obj is None:
#            return
#        self.info = getattr(obj, 'info', None)
#
#    def __reduce__(self):
#        # Get the parent's __reduce__ tuple
#        pickled_state = super().__reduce__()
#        # Create our own tuple to pass to __setstate__
#        new_state = pickled_state[2] + (self.info,)
#        # Return a tuple that replaces the parent's __setstate__ tuple with our own
#        return pickled_state[0], pickled_state[1], new_state
#
#    def __setstate__(self, state):
#        self.info = state[-1]  # Set the info attribute
#        # Call the parent's __setstate__ with the other tuple elements.
#        super().__setstate__(state[0:-1])
#
#    def __repr__(self):
#        return "{} info={}".format( super().__repr__(), self.info )
#
#    def __str__(self):
#        return "{} info={}".format( super().__str__(), self.info )

class Info(type):
    """Meta-class for parameters that adds an additional attribute *info*.

    *info* may be of arbitrary type."""

    def __new__(cls, name, bases, attrs):
        def infoNew(cls, info=None, *args, **kwargs ):
            obj = bases[0].__new__(cls,*args, **kwargs)
            if info is not None:
                obj.info = info
            return obj

        def infoArrayFinalize(self, obj):
            finalize = bases[0].__array_finalize__
            if inspect.isfunction(finalize):
                finalize(self,obj)
            if obj is not None:
                self.info = getattr(obj, 'info', None)

        def reduce(self):
            pickled_state = bases[0].__reduce__(self)
            new_state = pickled_state[2] + (self.info,)
            return pickled_state[0], pickled_state[1], new_state

        def setstate(self, state):
            self.info = state[-1]  # Set the info attribute
            bases[0].__setstate__(self,state[:-1])

        def repr(self):
            return "{} info={}".format( bases[0].__repr__(self), self.info )

        attrs['__new__'] = infoNew
        attrs['__array_finalize__'] = infoArrayFinalize
        attrs['__reduce__'] = reduce
        attrs['__setstate__'] = setstate
        attrs['__repr__'] = repr
        attrs['__str__'] = repr
        # assigning a mapping instead of a sequence of strings is allowed, but has no effect - probably provisioned for documentation. I find no way to assign info.__doc__
        attrs['__slots__'] = { 'info' : 'Arbitrary additional information' }
        return super(Info, cls).__new__(cls, name, bases, attrs)

class ADP(ParPar):
    """9-element vector marking the parameter block as camera distortion.

       Holds additional :attr:`~ParPar.parpar`\s *normalizationRadius* and *referencePoint*
    """
    __slots__ = ()
    @contract
    def __new__( subtype,
                 normalizationRadius : float,
                 referencePoint : adjust.AdpReferencePoint = adjust.AdpReferencePoint.principalPoint,
                 array = None ):
        "Initialize with given normalizationRadius and referencePoint. Optionally pass an array with initial values"
        # Mind that default-arguments are evaluated only once, at definition time.
        # Thus, all function calls share the same defaulted arguments, and if we don't copy `array` here,
        # then all instances of `ADP` would share the same underlying numpy.ndarray!
        # Also mind that different instances of `ADP` would still have their own IDs, even if they share their underlying numpy.ndarray.
        # To allow for externally referencing a passed np.ndarray, don't copy here, but default to None, and instantiate the actual default-ndarray in the function body.
        normalizationRadius = np.array([normalizationRadius], float)
        referencePoint = np.array([referencePoint], np.int32 ) # Note: np.dtype(np.int).itemsize==8 on Linux, and ==4 on Windows. In C++, always int32 is expected. So make it explicit here, too.
        return super().__new__(subtype, array, _FrozenDict( { "ADP.normalizationRadius":normalizationRadius, "ADP.referencePoint":referencePoint} ))

    name = 'Adp'
    staticNames = 'sk', 'scY', 'r3', 'r5', 't1', 't2', 'r7', 'r9', 'r11' # consider parpar

    @property
    @contract
    def normalizationRadius(self) -> float:
        "Radius of zero radial distortion, see :class:`~oriental.adjust._adjust.PhotoDistortion`."
        return self.parpar["ADP.normalizationRadius"][0].item()

    @normalizationRadius.setter
    @contract
    def normalizationRadius( self,
                             value : 'number,>0'):
        self.parpar["ADP.normalizationRadius"][0] = value

    @property
    @contract
    def referencePoint(self) -> adjust.AdpReferencePoint:
        "see :class:`~oriental.adjust._adjust.AdpReferencePoint`"
        # convert to enumeration type
        ret = self.parpar["ADP.referencePoint"][0].item()
        return adjust.AdpReferencePoint.values[ret]


    @referencePoint.setter
    @contract
    def referencePoint( self,
                        value : adjust.AdpReferencePoint ):
        self.parpar["ADP.referencePoint"][0] = value

#class ADP(_ADP):
#    __slots__ = 'parpar'

# Multiple inheritance here, see comment above.
# Info and _ADP must define empty slots, Info must be on the left of the list if base classes.
#class InfoADP(Info,_ADP):
#    __slots__ = 'info', 'parpar'

class InfoADP(ADP,metaclass=Info):
    pass

class EulerAngles(ParPar):
    """3-element vector marking the parameter block as rotation angles.

    Holds additional :attr:`~ParPar.parpar` *parametrization*."""
    __slots__ = ()
    @contract
    def __new__( subtype,
                parametrization : adjust.EulerAngles = adjust.EulerAngles.omfika,
                array = None ):
        "Initialize with given parametrization. Optionally pass an array with initial values"
        parametrization = np.array([parametrization], dtype=np.int32)
        return super().__new__(subtype, array, _FrozenDict( {'EulerAngles.parametrization':parametrization} ))

    name = 'Rot'
    staticNames = 'r1', 'r2', 'r3'

    @property
    def names(self):
        if self.parametrization == adjust.EulerAngles.alzeka:
            return '\N{GREEK SMALL LETTER ALPHA}', '\N{GREEK SMALL LETTER ZETA}', '\N{GREEK SMALL LETTER KAPPA}'
        if self.parametrization == adjust.EulerAngles.fiomka:
            return '\N{GREEK SMALL LETTER PHI}', '\N{GREEK SMALL LETTER OMEGA}', '\N{GREEK SMALL LETTER KAPPA}'
        return '\N{GREEK SMALL LETTER OMEGA}', '\N{GREEK SMALL LETTER PHI}', '\N{GREEK SMALL LETTER KAPPA}'


    # define a property as a handy short-cut for accessing parpar's "eulerAngles" using decorators
    @property
    @contract
    def parametrization(self) -> adjust.EulerAngles:
        "see :class:`~oriental.adjust._adjust.EulerAngles`"
        # convert to enumeration type
        ret = self.parpar["EulerAngles.parametrization"][0].item()
        return adjust.EulerAngles.values[ret]

    @parametrization.setter
    @contract
    def parametrization( self,
                         value : adjust.EulerAngles ):
        self.parpar["EulerAngles.parametrization"][0] = value

class InfoEulerAngles(EulerAngles,metaclass=Info):
    pass

class Ior(Array):
    "3-element vector marking the parameter block as interior orientation."
    __slots__ = ()
    name = 'Ior'
    staticNames = 'x0', 'y0', 'z0'

class InfoIor(Ior,metaclass=Info):
    pass

class Prc(Array):
    "3-element vector marking the parameter block as projection center position."
    __slots__ = ()
    name = 'Prc'
    staticNames = 'X0', 'Y0', 'Z0'

class InfoPrc(Prc,metaclass=Info):
    pass

class ObjectPoint(Array):
    "3-element vector marking the parameter block as object point."
    __slots__ = ()
    name = 'Obj'
    staticNames = 'X', 'Y', 'Z'

class InfoObjectPoint(ObjectPoint,metaclass=Info):
    pass

class TiePoint(ObjectPoint):
    "3-element vector marking the parameter block as tie object point."
    __slots__ = ()
    name = 'TP'

class InfoTiePoint(TiePoint,metaclass=Info):
    pass

class ControlPoint(ObjectPoint):
    "3-element vector marking the parameter block as ground control point."
    __slots__ = ()
    name = 'GCP'

class InfoControlPoint(ControlPoint,metaclass=Info):
    pass

class CheckPoint(ObjectPoint):
    "3-element vector marking the parameter block as (ground) check point."
    __slots__ = ()
    name = 'CP'

class InfoCheckPoint(CheckPoint,metaclass=Info):
    pass

new_contract('Ior',Ior)
new_contract('InfoIor',InfoIor)
new_contract('ADP',ADP)
new_contract('InfoADP',InfoADP)
new_contract('EulerAngles',EulerAngles)
new_contract('InfoEulerAngles',InfoEulerAngles)
new_contract('Prc',Prc)
new_contract('InfoPrc',InfoPrc)
new_contract('ObjectPoint',ObjectPoint)
new_contract('InfoObjectPoint',InfoObjectPoint)
new_contract('TiePoint',TiePoint)
new_contract('InfoTiePoint',InfoTiePoint)
new_contract('ControlPoint',ControlPoint)
new_contract('InfoControlPoint',InfoControlPoint)
new_contract('CheckPoint',CheckPoint)
new_contract('InfoCheckPoint',InfoCheckPoint)
