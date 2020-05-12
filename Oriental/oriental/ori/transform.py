# -*- coding: cp1252 -*-
from oriental import ori

from abc import abstractmethod
import functools
import collections.abc

from contracts import ContractsMeta, contract, new_contract
import numpy as np
from scipy import linalg
from osgeo import osr
osr.UseExceptions()

# contracts is just too slow ...
def contract( func ):
    return func

new_contract('CoordinateTransformation',osr.CoordinateTransformation)

class ITransform2D(metaclass=ContractsMeta):

    @abstractmethod
    @contract
    def inverted( self ):# -> ITransform2D: # doesn't work: ITransform2D not yet defined
        pass

    @abstractmethod
    @contract
    def forward( self,
                 pts : 'array[NxM] | array[M], M>1'
               ) -> 'array[NxM] | array[M], M>1':
        """1 point per row, or just 1 point as a vector"""
        pass

    @abstractmethod
    @contract
    def inverse( self,
                 pts : 'array[NxM] | array[M], M>1'
               ) -> 'array[NxM] | array[M], M>1':
        """"1 point per row, or just 1 point as a vector"""
        pass

    @abstractmethod
    @contract
    def meanScaleForward( self ) -> float:
        pass

new_contract('ITransform2D', ITransform2D )

class ITransform3D(metaclass=ContractsMeta):

    @abstractmethod
    @contract
    def inverted( self ):# -> ITransform3D: # doesn't work: ITransform3D not yet defined
        pass

    @abstractmethod
    @contract
    def forward( self,
                 pts : 'array[NxM] | array[M], M>2'
               ) -> 'array[NxM] | array[M], M>2':
        """1 point per row, or just 1 point as a vector"""
        pass

    @abstractmethod
    @contract
    def inverse( self,
                 pts : 'array[NxM] | array[M], M>2'
               ) -> 'array[NxM] | array[M], M>2':
        """"1 point per row, or just 1 point as a vector"""
        pass

    @abstractmethod
    @contract
    def meanScaleForward( self ) -> float:
        pass

new_contract('ITransform3D', ITransform3D )

class IdentityTransform2D(ITransform2D):

    def __init__( self ):
        pass

    @property
    def A( self ):
        return np.eye(2)

    @property
    def Ainv( self ):
        return np.eye(2)

    @property
    def t( self ):
        return np.zeros(2)

    def inverted( self ):
        return self

    def forward( self, pts ):
        return pts

    def inverse( self, pts ):
        return pts

    def meanScaleForward( self ):
        return 1.

class AffineTransform2D(ITransform2D):
    """" X_t = A*X_s + t """

    _Ainv = None

    @contract
    def __init__( self,
                  A : 'array[2x2]',
                  t : 'array[2]' ):
        self._A = A
        self.t = t

    @property
    def A( self ):
        return self._A

    @A.setter
    def A( self, value ):
        self._A = value
        self._Ainv = None

    @property
    def Ainv( self ):
        if self._Ainv is None:
            self._Ainv = linalg.inv(self.A)
        return self._Ainv

    def inverted( self ):
        return AffineTransform2D( self.Ainv, -self.Ainv.dot(self.t) )

    def forward( self, pts ):
        'left-multiply with the transpose in order to support pts with multiple rows'
        A = self.A.astype( pts.dtype )
        t = self.t.astype( pts.dtype )
        # use the ellipsis operator here to index into the last dimension
        res = pts[...,:2].dot( A.T ) + t
        return np.concatenate( ( res, pts[...,2:]), axis=pts.ndim-1 )

    def inverse( self, pts ):
        Ainv = self.Ainv.astype( pts.dtype )
        t    = self.t   .astype( pts.dtype )
        res = ( pts[...,:2] - t ).dot( Ainv.T )
        return np.concatenate( ( res, pts[...,2:] ), axis=pts.ndim-1 )

    def meanScaleForward( self ):
        return abs( linalg.det( self.A ) ) ** .5

    @staticmethod
    @contract
    def computeSimilarity( source : 'array[Nx2](float), N>1',
                           target : 'array[Nx2](float), N>1' ) -> '$AffineTransform2D':
        scale, R, x0, sigma = ori.similarityTrafo( source, target )
        return AffineTransform2D( scale*R, scale*R @ -x0 )

    @staticmethod
    @contract
    def computeAffinity( source : 'array[Nx2](float), N>1',
                         target : 'array[Nx2](float), N>1' ) -> '$AffineTransform2D':
        nPts = source.shape[0]
        A = np.zeros((nPts*2,6))
        A[0::2, 0] = source[:, 0]
        A[0::2, 1] = source[:, 1]
        A[0::2, 4] = 1.
        A[1::2, 2] = source[:, 0]
        A[1::2, 3] = source[:, 1]
        A[1::2, 5] = 1.
        b = np.zeros(nPts*2)
        b[:] = target.flat
        x, *_ = linalg.lstsq( A, b )

        return AffineTransform2D( A=x[:4].reshape((2,2)), t=x[4:] )

class AffineTransform3D(ITransform3D):
    """" X_t = A*X_s + t """

    _Ainv = None

    @contract
    def __init__( self,
                  A : 'array[3x3]',
                  t : 'array[3]' ):
        self._A = A
        self.t = t

    @property
    def A( self ):
        return self._A

    @A.setter
    def A( self, value ):
        self._A = value
        self._Ainv = None

    @property
    def Ainv( self ):
        if self._Ainv is None:
            self._Ainv = linalg.inv(self.A)
        return self._Ainv

    def inverted( self ):
        return AffineTransform3D( self.Ainv, -self.Ainv.dot(self.t) )

    def forward( self, pts ):
        """left-multiply with the transpose in order to support pts with multiple rows"""
        A = self.A.astype( pts.dtype )
        t = self.t.astype( pts.dtype )
        # use the ellipsis operator here to index into the last dimension
        res = pts[...,:3].dot( A.T ) + t
        return np.concatenate( ( res, pts[...,3:]), axis=pts.ndim-1 )

    def inverse( self, pts ):
        Ainv = self.Ainv.astype( pts.dtype )
        t    = self.t   .astype( pts.dtype )
        res = ( pts[...,:3] - t ).dot( Ainv.T )
        return np.concatenate( ( res, pts[...,3:] ), axis=pts.ndim-1 )

    def meanScaleForward( self ):
        return abs( linalg.det( self.A ) ) ** .5

    @staticmethod
    @contract
    def computeSimilarity( source : 'array[Nx3](float), N>2',
                           target : 'array[Nx3](float), N>2' ) -> '$AffineTransform3D':
        scale, R, x0, sigma = ori.similarityTrafo( source, target )
        return AffineTransform3D( scale*R, scale*R @ -x0 )

    @staticmethod
    @contract
    def computeAffinity( source : 'array[Nx3](float), N>2',
                         target : 'array[Nx3](float), N>2' ) -> '$AffineTransform3D':
        nPts = source.shape[0]
        A = np.zeros((nPts*3,12))
        A[0::3, 0] = A[1::3,  3] = A[2::3,  6] = source[:, 0]
        A[0::3, 1] = A[1::3,  4] = A[2::3,  7] = source[:, 1]
        A[0::3, 2] = A[1::3,  5] = A[2::3,  8] = source[:, 2]
        A[0::3, 9] = A[1::3, 10] = A[2::3, 11] = 1.
        b = np.zeros(nPts*3)
        b[:] = target.flat
        x, *_ = linalg.lstsq( A, b )

        return AffineTransform3D( A=x[:9].reshape((3,3)), t=x[9:] )

@functools.singledispatch
def transformPoints( trafo, pts ):
    return np.array( trafo.TransformPoints( np.atleast_2d(pts).tolist() ) )

@transformPoints.register(ITransform3D)
def _( trafo, pts ):
    return trafo.forward( pts )

@contract
def transformEOR( src2tgtCs : 'CoordinateTransformation|ITransform3D', images : collections.abc.Iterable, objPts = None ) -> None:
    for image in images:
        # How to transform rotations? PROJ.4 only supports the transformation of points!
        R = ori.euler2matrix(image.rot)
        # Offset by only 10. [m?] may lead to inaccurate results?
        offset = 10.
        pts = np.r_[ '0,2',
                     image.prc,
                     image.prc + offset * R[:,0], # R[:,0] is the direction of the camera's x-axis in object space, with the origin at the PRC.
                     image.prc + offset * R[:,1],
                     image.prc + offset * R[:,2] ]
        pts_tgtCs = transformPoints( src2tgtCs, pts )
        image.prc[:] = pts_tgtCs[0,:]
        R_tgtCs = pts_tgtCs[1:,:] - pts_tgtCs[0,:]
        R_tgtCs = R_tgtCs.T # pts holds the columns of R as rows!
        for idx in range(3):
            R_tgtCs[idx,:] /= linalg.norm(R_tgtCs[idx,:])
        rot = ori.matrix2euler( R_tgtCs )
        image.rot[:] = rot
        image.rot.parametrization = rot.parametrization

    if objPts is not None:
        for objPt in objPts:
            objPt[:] = transformPoints( src2tgtCs, objPt )
