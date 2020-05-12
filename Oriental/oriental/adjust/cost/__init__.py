# -*- coding: cp1252 -*-
"""
provides cost functions to be used by oriental.adjust.
"""

from ... import config as _config, _setup_summary_docstring

if _config.debug:
    from ._cost_d import *
    from ._cost_d import __date__
else:
    from ._cost import *
    from ._cost import __date__

import oriental.ori as _ori
from .. import EulerAngles
from contracts import contract
import numpy as np

class SimTrafo3d( AutoDiff ):
    @contract
    def __init__( self,
                  p : 'array[3](float)',
                  P : 'array[3](float)',
                  stdDevsP : 'array[3](float) | None' = None,
                  parametrization = EulerAngles.omfika ):
        #"p = s * R.T.dot( P - P0 )"
        "P = R.dot( p / s ) + P0" # compute the residuals in the global system, as usually, they are the ones that have a meaningful scale already -> make residuals interpretable.
        self.p = p
        self.P = P
        self.stdDevsP = stdDevsP
        self.parametrization = parametrization
        numResiduals = 3
        parameterSizes = (3,3,1) # P0, omfika, scale
        # initialize the base class
        super().__init__( numResiduals, parameterSizes )
        
    def Evaluate( self, parameters, residuals ):
        "Evaluate the ``residuals`` and possibly, their derivatives w.r.t. ``parameters``"

        P0 = parameters[0]
        if self.parametrization == EulerAngles.omfika:
            R = omFiKaToRotationMatrix( parameters[1] )
        elif self.parametrization == EulerAngles.fiomka:
            R = fiOmKaToRotationMatrix( parameters[1] )
        else:
            assert self.parametrization == EulerAngles.alzeka
            R = alZeKaToRotationMatrix( parameters[1] )
        scale = parameters[2][0]
        #residuals[:] = self.p - scale * R.T.dot( self.P - P0 )
        residuals[:] = self.P - ( R.dot( self.p / scale ) + P0 )

        if self.stdDevsP is not None:
            residuals[:] = residuals / self.stdDevsP
        # indicate success
        return True   

#class ObservedUnknown( AutoDiff ):
#    "Instead of ObservedUnknown, use adjust.cost.NormalPrior for multi-threaded evaluation"
#    @contract
#    def __init__( self,
#                  obs : 'array[N](float)',
#                  stdDevs : 'array[N](float) | None' = None ):
#        self.obs = obs.copy() # obs will be used as parameter, too, so make a copy
#        self.stdDevs = stdDevs
#        numResiduals = len(obs)
#        parameterSizes = len(obs),
#        super().__init__( numResiduals, parameterSizes )
#
#    def Evaluate( self, parameters, residuals ):
#        residuals[:] = self.obs - parameters[0]
#        if self.stdDevs is not None:
#            residuals[:] = residuals / self.stdDevs
#        return True   

class ObservedOmFiKa( AutoDiff ):
    @contract
    def __init__( self,
                  omFiKaLocal : 'array[3](float)',
                  omFiKaGlobal : 'array[3](float)',
                  omFiKaGlobalStdDevs : 'array[3](float) | None' = None ):
        self.RotLocal = _ori.omfika( omFiKaLocal )
        self.omFiKaGlobal = omFiKaGlobal
        self.omFiKaGlobalStdDevs = omFiKaGlobalStdDevs
        numResiduals = 3
        parameterSizes = 3,
        super().__init__( numResiduals, parameterSizes )

    def Evaluate( self, parameters, residuals ):
        omFiKaLocal2Global = parameters[0]
        dt = type(omFiKaLocal2Global[0])
        Rlocal2global = omFiKaToRotationMatrix( omFiKaLocal2Global )
        R = Rlocal2global.dot( self.RotLocal )
        estim = rotationMatrixToOmFiKa( R ) # for very small differential angles, we could use the simplified rotation matrix instead.
        #om = atan2( -R[1,2], R[2,2] )
        #fi = asin( max( -1., min( 1., R[0,2] ) ) ) # avoid a math domain error: argument of math.asin must be within [-1,1]
        #ka = atan2( -R[0,1], R[0,0] )
        #estim = np.array( [om,fi,ka], dtype=dt )
        residuals[:] = self.omFiKaGlobal - estim
        if self.omFiKaGlobalStdDevs is not None:
            residuals[:] = residuals / self.omFiKaGlobalStdDevs
        return True   

class LocalZDirection( AutoDiff ):
    @contract
    def __init__( self,
                  normal : 'array[3](float)',
                  stdDev : 'float|None' = None ):
        assert np.abs( 1. - np.sum(normal**2)**.5 ) < 1.e-7 # normal shall have unit length
        self.normal = normal
        self.stdDev = stdDev
        numResiduals = 1
        parameterSizes = 3,
        super().__init__( numResiduals, parameterSizes )

    def Evaluate( self, parameters, residuals ):
        R = omFiKaToRotationMatrix( parameters[0] )
        dt = type(parameters[0][0])
        # rotate the local unit normal vector by omfika and compute the cosine of its angle to the global z-axis.
        residuals[0] = acos( np.array([0.,0.,1.], dt).dot( R ).dot( self.normal ) ) * 200. / np.pi
        if self.stdDev is not None:
            residuals[0] = residuals[0] / self.stdDev
        return True   

class CameraRigPy( AutoDiff ):
    @contract
    def __init__( self,
                  offsetStdDevs : 'array[3](float) | None' = None,
                  omfikaStdDevs : 'array[3](float) | None' = None ):
        self.offsetStdDevs = offsetStdDevs
        self.omfikaStdDevs = omfikaStdDevs
        numResiduals = 6 # excenter, omfika
        parameterSizes = (3,)*6
        super().__init__( numResiduals, parameterSizes )

    def Evaluate( self, parameters, residuals ):
        ( offset,  # oblique img PRC in nadir img camCS
          omfika,
          P01,     # nadir img PRC in objCS
          omfika1, # nadir img ROT in objCS
          P02,     # oblique img PRC in objCS
          omfika2  # oblique img ROT in objCS
        ) = parameters

        R  = omFiKaToRotationMatrix( omfika )
        R1 = omFiKaToRotationMatrix( omfika1 )
        R2 = omFiKaToRotationMatrix( omfika2 )

        residuals[:3] = R1.T.dot( P02 - P01 ) - offset
        Rdiff = R2.T.dot( R1 ).dot( R )
        residuals[3:] = rotationMatrixToOmFiKa( Rdiff )

        if self.offsetStdDevs is not None:
            residuals[:3] /= self.offsetStdDevs
        if self.omfikaStdDevs is not None:
            residuals[3:] /= self.omfikaStdDevs

        return True   

def _summary():
    pass

_setup_summary_docstring( _summary, __name__ )