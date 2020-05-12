# -*- coding: cp1252 -*-
"Provides types to organize a block"

from contracts import contract
# support **compactObj (double-asterisk) for initialisation of derived Compacts from base-Compacts:
# ImageWKeyPts = blocks.types.createCompact( ImageWId, ('keypts',) )
# ImageWKeyPts( keypts=np.array( value ), **images[imgId] )
import collections.abc

class Camera:
    """Base type for cameras.
       Interior orientation (ior) and distortion parameters (adp) constitute the minimal set of attributes.
       As ior, pass |eg| :class:`~oriental.adjust.parameters.Ior`.
       As adp, pass |eg| :class:`~oriental.adjust.parameters.ADP`.
    """
    __slots__ = 'ior', 'adp'
    @contract
    def __init__(self, ior : 'array[3](float)',
                       adp : 'array[9](float)'):
        self.ior=ior
        self.adp=adp

class Image:
    """Base type for images.
       Projection center position (prc) and Euler rotation angles (rot) constitute the minimal set of attributes.
       As prc, pass |eg| :class:`~oriental.adjust.parameters.Prc`.
       For rot, pass |eg| :class:`~oriental.adjust.parameters.EulerAngles`.
    """
    __slots__ = 'prc', 'rot'
    @contract
    def __init__(self, prc : 'array[3](float)',
                       rot : 'array[3](float)'):
        self.prc=prc
        self.rot=rot

class Info:
    "Base class for attaching information to types."
    __slots__= ()

# module singleton. Note that this won't be pickled.
_klasses = {}

def instantiateCompact( *bases ):
    "restores classes created with :func:`createCompact`, adding pickle support"
    klass = bases[0]
    for attrib in bases[1:]:
        klass = createCompact(klass,attrib)
    return klass.__new__(klass)

@contract
def createCompact( Base : type, attribs : 'str|seq(str)' ) -> type:
    """returns a class derived from Base with `attribs` as additional attribute(s)
       If Base has no __dict__, then the returned class will not have one, either.
       The returned class is pickleable.
       Base may be the result of a preceding call to createCompact, thereby supporting inheritance.
    """
    if isinstance( attribs, str ):
        attribs = (attribs,)
    else:
        attribs = tuple(attribs)
    klass = _klasses.get((Base,attribs))
    if klass is not None:
        return klass
    
    # ensure that collections.abc.Mapping is derived from only once, even for Compacts derived from another Compact
    if issubclass(Base, collections.abc.Mapping):
        bases = (Base,)
    else:
        bases = (Base,collections.abc.Mapping)

    class Compact(*bases):
        __slots__ = attribs
        def __init__(self,**kwargs):
            assert len(set(Compact.allSlots)) == len(Compact.allSlots), "Base class slot names overlap with own slot names"
            #  Make sure that all own slots are assigned (default=None).
            #  Pass the rest to the base-class-__init__, which may define defaults, type checks, etc.
            for k in Compact.__slots__:
                setattr(self,k,kwargs.pop(k,None))
            Base.__init__(self,**kwargs)
        # support pickling, even though this class is created dynamically
        def __reduce__(self):
            # as the callable (first arg), return a function and not a class instance, because that class instance would be pickled, too,
            # resulting in different Compact-classes before and after pickle:
            # orig.__class is unpickled.__class__ # would be False
            #res = instantiateCompact, (Base,attribs), tuple(getattr(self,k) for k in Compact.allSlots )
            # Support pickling of Compact classes derived from another Compact class.
            # n.b.: everything returned from __reduce__ must be picklable, and hence we must not return Compact.__base__ if that is a Compact-class, too!
            # Instead, return a tuple of arguments that allow instantiateCompact to recursively reconstruct the class.
            base = Base
            args = [attribs]
            while (base.__module__, base.__qualname__) == (Compact.__module__, Compact.__qualname__):
                args.append(base.__slots__)
                base = base.__base__
            args.append( base )
            args.reverse()
            res = instantiateCompact, tuple(args), tuple(getattr(self,k) for k in Compact.allSlots )
            return res
        def __setstate__(self,state):
            for k,v in zip(Compact.allSlots,state):
                setattr(self,k,v)
        def __repr__(self):
            return "{} {}".format( Base.__name__, ' '.join('{}={}'.format( k, getattr(self,k) ) for k in Compact.allSlots ) )
        __str__ = __repr__
        __doc__ = Base.__name__ + ' with ' + ', '.join(attribs)
        # iterator protocol support, dict-like
        def __getitem__(self, key):
            return getattr(self,key)
        def __iter__(self):
            for slot in Compact.allSlots:
                yield slot
        def __len__(self):
            return len(Compact.allSlots)
    Compact.allSlots = [ slot for klass in reversed(Compact.mro()) for slot in getattr(klass,'__slots__',()) ]
    Compact.__name__ = Base.__name__ + ''.join(el.capitalize() for el in attribs) # make error messages readable: "'CameraId' object has no attribute ..." instead of "'Compact' object ..."

    _klasses[Base,attribs] = Compact
    return Compact

InfoId = createCompact( Info, 'id' )
InfoIdStdDevs = createCompact( InfoId, 'stdDevs' )
