# -*- coding: cp1252 -*-
"""
one- and two-view geometry
"""

from .. import config

if config.debug:
    from ._ori_d import *
    from ._ori_d import __date__
else:
    from ._ori import *
    from ._ori import __date__

# import types
# import functools
# def wrapAsPythonFunction(func):
#    @functools.wraps(func)
#    def wrapper(*args, **kwargs):
#        return func( *args, **kwargs )
#    return wrapper
#
# for name, attr in tuple(globals().items())[:]:
#    if isinstance(attr, types.BuiltinFunctionType):
#        globals()[name] = wrapAsPythonFunction(attr)
#    elif isinstance(attr, type):
#        for name2, attr2 in tuple(vars(attr).items())[:]:
#            if isinstance(attr2, types.BuiltinMethodType):
#                setattr(attr, name2, wrapAsPythonFunction(attr2))

from .. import utils
from oriental.adjust import EulerAngles as _EulerAngles
from oriental.adjust.parameters import EulerAngles as _EulerAnglesPar, ADP as _ADP

from contracts import contract
import numpy as np
from scipy import linalg
import math
import cv2

import scipy.special


@contract
def r2d(angle: float) -> 'array[2x2](float)':
    angle *= np.pi / 200.
    san = np.sin(angle)
    can = np.cos(angle)
    # rotate the object, not the coordinate system (in a mathematically positive sense)
    return np.array([[can, -san],
                     [san, can]])


@contract
def rx(angle: float) -> 'array[3x3](float)':
    res = np.eye(3)
    res[1:, 1:] = r2d(angle)
    return res


@contract
def ry(angle: float) -> 'array[3x3](float)':
    res = np.eye(3)
    res[[0, 0, 2, 2], [0, 2, 0, 2]] = r2d(angle).T.flat
    return res


@contract
def rz(angle: float) -> 'array[3x3](float)':
    res = np.eye(3)
    res[:2, :2] = r2d(angle)
    return res


# omfika([om,fi,ka])   == rx( om) @ ry( fi) @ rz( ka)
# omfika([om,fi,ka]).T == rz(-ka) @ ry(-fi) @ rx(-om)

@contract
def maxNumItersRANSAC(nModelPoints: int,
                      nDataPoints: int,
                      inlierRatio: 'float,>0,<1',
                      confidence: 'float,>0,<1') \
        -> int:
    # Laut 1981 Fischler,Bolles - Random Sample Concensus - ACM können wir aber vorab die nötige Maximalanzahl an Iterationen schätzen, auf Basis der
    # - Wahrscheinlichkeit w, dass ein beliebiger Punkt ein inlier ist,
    # - der Anzahl n der zur Schätzung der Modellparameter minimal benötigten Samples
    # - der gewünschten probability z, dass zumindest eine Stichprobe aus lauter inliern besteht.
    # Die max. Anzahl an Iterationen k ergibt sich dann zu:
    # k = [log(1 - z)]/[log(1 - b)],
    # wobei
    # b = w^n
    # siehe cv::RANSACUpdateNumIters

    # Beispiel mit
    # n=4 (cv2.solvePnPRansac)
    # 10% inliers -> 69074       @ 99.9% probability of at least 1 correct model
    #  8% inliers -> 168643      @ 99.9% probability of at least 1 correct model
    #  5% inliers -> 1.1 million @ 99.9% probability of at least 1 correct model
    b = inlierRatio ** nModelPoints
    k = math.log(1. - confidence) / math.log(1. - b)

    # Assuming that RANSAC selects a different sample each time,
    #   it doesn't make sense to execute more iterations than there are possibilities to choose different sample point sets from the data.
    # OpenCV implementations of RANSAC do not ensure that a different sample is taken each time (they do not consider previous selections when selecting a sample).
    # However, it might still be a good idea to use that upper bound, which affects the result only if the data set is small, and the inlierRatio is very low.
    # Note: with exact=False, comb returns a 1-element, 0-dim array.
    #            exact=True,  comb returns a scalar!
    # -> Use .item() to convert to a scalar 
    return int(min(math.ceil(k), scipy.special.comb(N=nDataPoints, k=nModelPoints, exact=False).item()))

    # return int(math.ceil(k))


@contract
def similarityTrafo(x: 'array[RxC](float),(R=C|R>C),(C=2|C=3)',
                    y: 'array[RxC](float),(R=C|R>C),(C=2|C=3)') \
        -> 'tuple(float,array[CxC](float),array[C](float),float),(C=2|C=3)':
    """Compute a 2-D or 3-D similarity transformation such that y=s*R.dot(x-x0)
    has the minimum mean squared error
    
    Photogr.Manual 5th ed. p828"""

    dim = x.shape[1]

    x = x.T
    y = y.T
    n_points = x.shape[1]

    # form: y-yo=s*R(x-xo)
    y0 = np.mean(y, axis=1)
    x0 = np.mean(x, axis=1)

    dy = (y.T - y0).T
    dx = (x.T - x0).T

    # s1=0;
    # s2=0;
    # H=np.zeros((dim,dim))
    # for i in range(n_points):
    #    s1 += dy[:,i].dot( dy[:,i] )
    #    s2 += dx[:,i].dot( dx[:,i] )
    #    H += np.outer( dx[:,i], dy[:,i] )

    # H=np.zeros((dim,dim))
    # for ix in range(dim):
    #    for iy in range(dim):
    #        H[ix,iy] = dx[ix,:].dot( dy[iy,:] )
    H = dx.dot(dy.T)

    s1 = np.sum(dy ** 2.)
    s2 = np.sum(dx ** 2.)

    s = (s1 / s2) ** 0.5
    # note: S is a vector; V is V.T
    [U, S, V] = linalg.svd(H)
    R = V.T.dot(U.T)

    # check sign of R; 2013-03-29 (not fully tested)
    if linalg.det(R) < 0:
        if linalg.det(V.T) < 0:
            V.T[:, -1] *= -1.
        elif linalg.det(U) < 0:
            U[:, -1] *= -1.

        R = V.T.dot(U.T)

    # form: y=s*R(x-xo)
    x0 -= 1 / s * R.T.dot(y0)

    # residuals
    sigma0 = 0.
    if n_points > dim:
        y2 = s * R.dot((x.T - x0).T)
        # for i in range(n_points):
        #    d = y2[:,i] - y[:,i]
        #    res += d.dot( d )

        resSqr = np.sum((y2 - y) ** 2)
        unk = 7 if dim == 3 else 4
        sigma0 = (resSqr / (dim * n_points - unk)) ** .5

    return s, R, x0, sigma0


""" extract Euler angles [gon]: omega, phi, kappa
 Achtung auf unterschiedliche Definitionen!

 oriental.adjust verwendet ORIENT-Definition (R transponiert!):
 (p-p_0) = s*R'*(P-P_0)
 wobei das KS, und nicht das Objekt gedreht wird -> z.B.:
 R_x(\omega) = [ 1       0           0
                 0  cos(\omega) sin(\omega)
                 0 -sin(\omega) cos(\omega) ]
 Aber: ist im Orpheus-manual, 14.12.1 wirklich R_{\omega,\phi,\kappa} angegeben, oder dessen Transponierte?
 Beachte: (A*B)' = B'*A' !

 openCV: anderes Bildkoordinatensystem: Ursprung zwar auch links oben, aber: x nach rechts, y nach unten, z daher weg vom PRC!
 -> Bildkoordinaten, Objektkoordinaten, Translation, Rotation entsprechend transformieren!
 x = A [R|t] X

 Achtung: bei Orient wird P0 abgezogen, bei openCV wird t addiert!"""


@contract
def projectionMat2oriRotTrans(P: 'array[3x4](float)'):
    # rotation about the x-axis by 200 gon
    Rx200 = np.diag((1., -1., -1.))
    R = Rx200.T @ P[:, :3] @ Rx200
    # R in ORIENT is the rotation from the camera CS to the object CS
    R = R.T
    # openCV uses R(x)+t, ORIENT uses R(x-t)
    t = np.dot(Rx200.T, np.dot(P[:, :3].T, -P[:, 3]))
    return R, t


@contract
def oriRotTrans2projectionMat(R: 'array[3x3](float)',
                              t: 'array[3](float)'):
    Rx200 = np.diag((1., -1., -1.))

    P = np.empty((3, 4))
    P[:, :3] = Rx200 @ R.T @ Rx200.T
    P[:, 3] = - P[:, :3] @ Rx200 @ t
    return P


@contract
def omfika_py(arg: 'array[3](float) | array[3x3](float)') -> 'array[3x3](float) | array[3](float)':
    if arg.shape == (3,):  # convert to rotation matrix
        omfika_ = arg
        om = omfika_[0] / 200. * np.pi
        fi = omfika_[1] / 200. * np.pi
        ka = omfika_[2] / 200. * np.pi

        som = math.sin(om)
        com = math.cos(om)
        sfi = math.sin(fi)
        cfi = math.cos(fi)
        ska = math.sin(ka)
        cka = math.cos(ka)

        # as in ORPHEUS manual 14.12.1
        # This is R, not R.T !
        R = np.array([[cfi * cka, -cfi * ska, sfi],
                      [com * ska + som * sfi * cka, com * cka - som * sfi * ska, -som * cfi],
                      [som * ska - com * sfi * cka, som * cka + com * sfi * ska, com * cfi]])
        return R

    if arg.shape == (3, 3):  # convert to rotation angles ...
        R = arg
        om = math.atan2(-R[1, 2], R[2, 2])
        # avoid a math domain error: argument of math.asin must be within [-1,1]
        fi = math.asin(max(-1., min(1., R[0, 2])))
        ka = math.atan2(-R[0, 1], R[0, 0])
        omfika_ = np.array([om, fi, ka])
        return omfika_ * 200 / np.pi

    raise Exception("3-vector or rotation matrix expected, shape is: {}".format(arg.shape))


@contract
def omfika(arg: 'array[3](float) | array[3x3](float)') -> 'array[3x3](float) | array[3](float)':
    if arg.shape == (3,):  # convert to rotation matrix
        return omFiKaToRotationMatrix(arg)
    if arg.shape == (3, 3):  # convert to rotation angles ...
        arg = np.ascontiguousarray(arg)  # make C-contiguous (row-major order)
        return _EulerAnglesPar(_EulerAngles.omfika, rotationMatrixToOmFiKa(arg))
    raise Exception("3-vector or rotation matrix expected, but shape is: {}".format(arg.shape))


@contract
def fiomka(arg: 'array[3](float) | array[3x3](float)') -> 'array[3x3](float) | array[3](float)':
    if arg.shape == (3,):  # convert to rotation matrix
        return fiOmKaToRotationMatrix(arg)
    if arg.shape == (3, 3):  # convert to rotation angles ...
        arg = np.ascontiguousarray(arg)
        return _EulerAnglesPar(_EulerAngles.fiomka, rotationMatrixToFiOmKa(arg))
    raise Exception("3-vector or rotation matrix expected, but shape is: {}".format(arg.shape))


@contract
def alzeka(arg: 'array[3](float) | array[3x3](float)') -> 'array[3x3](float) | array[3](float)':
    if arg.shape == (3,):  # convert to rotation matrix
        return alZeKaToRotationMatrix(arg)
    if arg.shape == (3, 3):  # convert to rotation angles ...
        arg = np.ascontiguousarray(arg)
        return _EulerAnglesPar(_EulerAngles.alzeka, rotationMatrixToAlZeKa(arg))
    raise Exception("3-vector or rotation matrix expected, but shape is: {}".format(arg.shape))


@contract
def euler2matrix(eulerAngles: 'array[3](float)') -> 'array[3x3](float)':
    parametrization = getattr(eulerAngles, 'parametrization', None)
    if parametrization == _EulerAngles.fiomka:
        return fiomka(eulerAngles)
    elif parametrization == _EulerAngles.alzeka:
        return alzeka(eulerAngles)
    return omfika(eulerAngles)


@contract
def matrix2euler(R: 'array[3x3](float)') -> 'array[3](float)':
    if np.abs(np.arccos(np.clip(R[2, 2], -1., 1.))) < np.pi / 4:
        return omfika(R)
    return alzeka(R)


@contract
def ropiyaw(arg: 'array[3](float) | array[3x3](float)') -> 'array[3x3](float) | array[3](float)':
    if arg.shape == (3, 3):  # extract angles in gon
        return -omfika(arg.T)

    return omfika(-arg).T


@contract
def cameraMatrix(ior: 'array[3](float)') -> 'array[3x3](float)':
    return np.array([[ior[2], 0., ior[0]],
                     [0., ior[2], -ior[1]],
                     [0., 0., 1.]])


# @contract
# def hardwareExif2Rot( exifRot : dict ) -> 'array[3x3](float)':
#    "returns a rotation matrix w.r.t. the local north and gravity directions"
#
#    # angles in Exif are in degrees!
#    om = exif['GPS_0xd001']      / 180. * np.pi
#    fi = exif['GPS_0xd000']      / 180. * np.pi
#    ka = exif['GPSImgDirection'] / 180. * np.pi
#
#    som = math.sin(om)
#    com = math.cos(om)
#    sfi = math.sin(fi)
#    cfi = math.cos(fi)
#    ska = math.sin(ka)
#    cka = math.cos(ka)
#
#    R_om = np.array([ [   1.,  0.,   0. ],
#                      [   0., com, -som ],
#                      [   0., som,  com ] ] )
#
#    R_fi  = np.array([ [  cfi,  0.,  sfi ],
#                       [   0.,  1.,   0. ],
#                       [ -sfi,  0.,  cfi ] ] )
#
#    R_ka = np.array([ [ cka, -ska,  0. ],
#                      [ ska,  cka,  0. ],
#                      [  0.,   0.,  1. ] ] )
#
#    R = R_ka.dot( R_fi ).dot( R_om )
#
#    R = R.dot( np.array([[  0., +1., 0. ],
#                         [ -1.,  0., 0. ],
#                         [  0.,  0., 1. ]])  # rot about z by +90°
#        ).dot( np.array([[  1.,  0.,  0. ],
#                         [  0.,  0.,  -1. ],
#                         [  0., +1.,  0. ]]) ) # rot about x by -90°
#    return R

@contract
def adpParam2Struct(adp: _ADP):
    # here we map to the ORIENT IDs!
    adp_ = np.empty((len(adp) + 2, 2))
    adp_[0, 0] = -1;
    adp_[0, 1] = int(adp.referencePoint)
    adp_[1, 0] = 0;
    adp_[1, 1] = adp.normalizationRadius
    adp_[2:8, 0] = np.arange(1, 7)
    adp_[8:, 0] = np.arange(37, 40)
    adp_[2:, 1] = adp
    return adp_


@contract
def projection(P: 'array[3](float)|array[Nx3](float)',
               PRC: 'array[3](float)',
               eulerAnglesOrR: 'array[3](float)|array[3x3](float)',
               ior: 'array[3](float)',
               adp=None) -> 'array[2](float)|array[Nx2](float)':
    """compute the residual according to ORIENT"""
    # (p-p_0) = s*R'*(P-P_0)
    if eulerAnglesOrR.ndim == 2:
        R = eulerAnglesOrR
    else:
        R = euler2matrix(eulerAnglesOrR)
    Xred = P - PRC
    # as in ORPHEUS manual 14.3.7
    # note: as we use R.T, we use columns of R instead of rows
    denominator = Xred @ R[:, 2]
    x = ior[0] - ior[2] * (Xred @ R[:, 0] / denominator)
    y = ior[1] - ior[2] * (Xred @ R[:, 1] / denominator)
    imgPts = np.ascontiguousarray(np.column_stack([x, y]))  # make C-contiguous
    if adp is not None and adp.any():
        adp_ = adpParam2Struct(adp)
        distortion_inplace(imgPts, ior, adp_, DistortionCorrection.dist)
    return np.squeeze(imgPts)


@contract
def get_epipole(F: 'array[3x3](float)'):
    u, s, v = linalg.svd(F);
    # Achtung!
    # x = u*diag(s)*v
    # nicht: u*diag(s)*v.T
    # d.h. v ist hier die Transponierte vom v in Matlab
    e = v.T[:, -1]  # it's the right null vector of F_0
    eprime = u[:, -1]
    return e, eprime


@contract
def crossProductMatrix(x: 'array[3](float)'):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


@contract
def rodrigues(R: 'array[3](float) | array[3x3](float)'):
    # does the same as cv2.Rodrigues
    if R.shape == (3, 3):
        # see http://en.wikipedia.org/wiki/Axis_angle#Log_map_from_SO.283.29_to_so.283.29
        theta = math.acos((np.trace(R) - 1.) / 2.)
        if abs(theta) < 1.e-10:
            return np.zeros(3)

        axis = 1. / (2 * math.sin(theta)) * np.array([R[2, 1] - R[1, 2],
                                                      R[0, 2] - R[2, 0],
                                                      R[1, 0] - R[0, 1]])
        return axis * theta

    if R.shape == (3,):
        # see http://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Conversion_to_rotation_matrix
        # R: 3-vector; the rotation angle is encoded as length in [rad]
        eye = np.eye(3, dtype=np.float64)

        k = np.array(R, dtype=np.float64)
        theta = linalg.norm(k)
        if abs(theta) < 1.e-10:
            return eye
        k /= theta

        kkt = np.outer(k, k)
        skew = crossProductMatrix(k)

        mtx = kkt + np.cos(theta) * (eye - kkt) + np.sin(theta) * skew

        # mtx2 = eye * np.cos(theta) + skew * np.sin(theta) + ( 1 - np.cos(theta) ) * kkt
        return mtx

    raise Exception(
        "R is expected to be either a 3x3 rotation matrix, or a 3-vector whose length encodes the rotation angle")


# pt1,pt2: (distorted) image observations in ORIENT camera coordinate system
# returns triangulated points in ORIENT model coordinate system
# Note: cv2.triangulatePoints uses SVD (linear triangulation). Thus, results may deviate from the least-squares solution
# For the least-squares solution, openCV seems to offer only cvCorrectMatches, which expects an F-matrix, (which may be derived from PRC and ROT)
@contract
def triangulatePoints(pt1: 'array[Nx2](float)|array[2](float)',
                      pt2: 'array[Nx2](float)|array[2](float)',
                      camParams1,
                      camParams2
                      ) -> 'array[Nx3](float)':
    P1 = oriRotTrans2projectionMat(camParams1.R, camParams1.t)
    P2 = oriRotTrans2projectionMat(camParams2.R, camParams2.t)

    K1 = cameraMatrix(camParams1.ior)
    K2 = cameraMatrix(camParams2.ior)
    KInv1 = linalg.inv(K1)
    KInv2 = linalg.inv(K2)

    # pt1_undist: undistorted image points in OpenCV image CS
    pt1_undist = distortion(np.atleast_2d(pt1), camParams1.ior, adpParam2Struct(camParams1.adp),
                            DistortionCorrection.undist) * (1., -1.)
    pt2_undist = distortion(np.atleast_2d(pt2), camParams2.ior, adpParam2Struct(camParams2.adp),
                            DistortionCorrection.undist) * (1., -1.)

    pth1 = np.hstack((pt1_undist, np.ones((pt1_undist.shape[0], 1)))).T
    pth2 = np.hstack((pt2_undist, np.ones((pt2_undist.shape[0], 1)))).T
    pt_normalized1 = KInv1.dot(pth1)
    pt_normalized2 = KInv2.dot(pth2)
    pt_normalized1 = pt_normalized1[:2, :] / pt_normalized1[2, :]
    pt_normalized2 = pt_normalized2[:2, :] / pt_normalized2[2, :]

    # Achtung: entweder:
    # P=[R|t] und pt ist normalized (==K^-1.dot(pt)), oder
    # P=K[R|t] und pt sind nicht normalisiert.
    # Die Ergebnisse sollten die gleichen sein. Tlw. sind sie aber sehr unterschiedlich (für Ausreißer?)
    X = cv2.triangulatePoints(P1, P2, pt_normalized1, pt_normalized2)
    X[:3, :] /= X[3, :]

    # transform object coordinates to ORIENT system: invert both y- and z-coordinates
    Xori = X[:3, :].T
    Xori[:, 1:] *= -1.

    return Xori


# Hartley&Zisserman, S. 312
# cv2.triangulatePoints setzt pro Punkt und Bild 1 zusätzliche Gleichung an! Diese ist lt. H&Z linear abhängig. Die Ergebnisse sind aber trotzdem besser, scheint es.
@contract
def triangulate(P1: 'array[3x4](float)',
                P2: 'array[3x4](float)',
                x1: 'array[NxM](float) | array[N](float),((N=2)|(N=3))',
                x2: 'array[NxM](float) | array[N](float), ((N=2)|(N=3))',
                homogeneous: bool = True, normalize=True) \
        -> 'array[3xM](float) | array[3](float)':
    if x1.ndim == 1:
        x1 = x1[np.newaxis, :].T
    if x2.ndim == 1:
        x2 = x2[np.newaxis, :].T
    if x1.shape[0] == 3:
        assert np.all(x1[2, :] == 1.) and np.all(x2[2,
                                                 :] == 1.), 'x1 must either contain Cartesian 2-d points columnwise, or homogeneous 2-d points columnwise with the last coordinate being 1'
    Xhat = np.ones((4, x1.shape[1]))
    # P1 = np.hstack( (np.eye(3), np.zeros((3, 1))) )
    for i in range(x1.shape[1]):
        A = np.array([x1[0, i] * P1[2, :] - P1[0, :],
                      x1[1, i] * P1[2, :] - P1[1, :],
                      x2[0, i] * P2[2, :] - P2[0, :],
                      x2[1, i] * P2[2, :] - P2[1, :]])
        # scale the columns to have unit-L2-norm
        if normalize:
            A = (A.T / np.sum(A ** 2, axis=0) ** .5).T
            # for iCol in range(A.shape[1]):
            #    A[:, iCol] /= linalg.norm(A[:, iCol])
        if homogeneous:
            U, s, Vt = linalg.svd(A)
            Xhat[:, i] = Vt.T[:, -1]
        else:
            b = -A[:, -1]
            A = A[:, :-1]
            U, s, Vt = linalg.svd(A)
            bprime = U.T @ b
            y = bprime[:3] / s
            Xhat[:3, i] = Vt.T @ y

    Xhat /= Xhat[3, :]

    if 0:
        x1_check = P1 @ Xhat
        x1_check /= x1_check[2, :]
        x2_check = P2 @ Xhat
        x2_check /= x2_check[2, :]
        # okay: x1 ~= x1_check, x2 ~= x2_check

    return Xhat[:3, :].squeeze()


@contract
def maxReprojectionErrorSquared(x1: 'array[Nx2](float)',
                                x2: 'array[Nx2](float)',
                                F: 'array[3x3](float)') \
        -> float:
    # x1,x2: zeilenweise nicht-homologe Bildkoordinaten
    # F: Fundamentalmatrix

    # Implementierung analog zu openCV's CvFMEstimator::computeReprojError
    # ergibt das Maximum der quadrierten Residuen in Bild 1 und Bild 2
    ones = np.ones((x1.shape[0], 1))

    # [3,n]
    x1h = np.vstack((x1.T, ones.T))
    x2h = np.vstack((x2.T, ones.T))

    # [3,n]
    abc = np.dot(F, x1h)

    err2_2 = np.sum(x2h * abc, axis=0)
    err2_2 *= err2_2 / (abc[0, :] ** 2 + abc[1, :] ** 2)

    abc = np.dot(F.T, x2h)

    err1_2 = np.sum(x1h * abc, axis=0)
    err1_2 *= err1_2 / (abc[0, :] ** 2 + abc[1, :] ** 2)

    err_2 = np.maximum(err1_2, err2_2)

    return err_2


@contract
def recoverPose2(essentialMatrix: 'array[3x3](float)',
                 ptNormalized1: 'array[Nx2](float)',
                 ptNormalized2: 'array[Nx2](float)') \
        -> 'tuple( int, array[3x3](float), array[3](float) )':
    P1 = np.hstack((np.eye((3)), np.zeros((3, 1))))
    R1, R2, t = cv2.decomposeEssentialMat(essentialMatrix)

    def getP2(idx):
        if idx == 0:
            return np.hstack((R1, t))
        elif idx == 1:
            return np.hstack((R1, -t))
        elif idx == 2:
            return np.hstack((R2, t))
        elif idx == 3:
            return np.hstack((R2, -t))
        else:
            raise Exception("index out of range")

    nGood = np.zeros(4, dtype=np.uint64)
    for idx in range(4):
        P2 = getP2(idx)
        X = cv2.triangulatePoints(P1, P2, ptNormalized1.T, ptNormalized2.T)
        x2 = np.dot(P2, X)
        X = X[:3, :] / X[3, :]
        nGood[idx] = (np.logical_and(X[2] > 0., x2[2] > 0.)).sum()

    iMax = np.argmax(nGood)
    P2 = getP2(iMax)
    ret = nGood.item(iMax), P2[:, :3], P2[:, 3].reshape((-1, 1))

    if 1:  # check with openCV. Only valid if points at 'infinity' do not influence the selection of R2 and t2
        if ret[0] > 0.5 * ptNormalized1.shape[0]:
            nGoodCv, R2Cv, t2Cv, _ = cv2.recoverPose(essentialMatrix, ptNormalized1, ptNormalized2)
            if ret[0] < nGoodCv \
                    or not np.allclose(ret[1], R2Cv) \
                    or not np.allclose(ret[2], t2Cv):
                dummy = 0  # assert False

    return ret


@contract
def recoverPose(essentialMatrix: 'array[3x3](float)',
                ptNormalized1: 'array[2xN](float)',
                ptNormalized2: 'array[2xN](float)') \
        -> 'array[3x4](float)':
    """recover R and t from the given essentialMatrix, using the passed image observations
       to check which combination of R and t results in object points that are located
       in front of both cameras
       ptNormalized1,ptNormalized2: image observations that have been normalized to a focal length==1 and a principal point at (0,0)
       alternative: use cv2.recoverPose
       drawback of cv2.recoverPose: it uses a hard-coded threshold for maximum object point distances, beyond which,
       object points are considered to lie close to infinity, and thus, no decision can be made if they lie in front of or behind the camera
       -> they are excluded from the decision.
       However, in aerial photogrammetry, a base-to-height ratio greater than 50 is no exception!"""

    P1 = np.hstack((np.eye((3)), np.zeros((3, 1))))

    # Achtung! überraschende Implementierung von linalg.svd:
    # u,s,v = linalg.svd(E)
    # E == u*diag(s)*v
    # nicht: E = u*diag(s)*v.T
    U, S, V = linalg.svd(essentialMatrix);

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]], dtype=np.float64);

    # !!! http://tech.groups.yahoo.com/group/OpenCV/message/52820
    #  "When a det(R)=-1 You must multiply the essential matrix by -1 and everything is correct."
    # bzw. http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid - last paragraph:
    # "Third, it must also need to be shown that the above expression for R is a rotation matrix. It is the product of three matrices which all are orthogonal which means that R, too, is orthogonal or det(R) = 1.
    #   To be a proper rotation matrix it must also satisfy det(R) = 1 . Since, in this case, E is seen as a projective element this can be accomplished by reversing the sign of E if necessary."                  
    # if linalg.det( np.dot( U, np.dot( W   , V ) ) ) < 0:
    #   essentialMatrix *= -1.
    #   U,S,V = linalg.svd(essentialMatrix)
    # https://github.com/jesolem/PCV/blob/master/PCV/geometry/sfm.py:140
    if linalg.det(np.dot(U, V)) < 0:
        V = -V

    if 0:
        # check that only 1 out of the 4 possible combinations results in object points that are located in front of both cameras
        # This check is insecure, because the points used for the check may be blunders.
        P2_1 = np.hstack((np.dot(U, np.dot(W, V)), U[:, -1].reshape((-1, 1))))
        P2_2 = np.hstack((np.dot(U, np.dot(W, V)), -U[:, -1].reshape((-1, 1))))
        P2_3 = np.hstack((np.dot(U, np.dot(W.T, V)), U[:, -1].reshape((-1, 1))))
        P2_4 = np.hstack((np.dot(U, np.dot(W.T, V)), -U[:, -1].reshape((-1, 1))))

        check_pt_normalized1 = ptNormalized1[:, :5]
        check_pt_normalized2 = ptNormalized2[:, :5]

        X1 = triangulate(check_pt_normalized1, check_pt_normalized2, P2_1, True)
        x2_1 = np.dot(P2_1, X1)
        X2 = triangulate(check_pt_normalized1, check_pt_normalized2, P2_2, True)
        x2_2 = np.dot(P2_2, X2)
        X3 = triangulate(check_pt_normalized1, check_pt_normalized2, P2_3, True)
        x2_3 = np.dot(P2_3, X3)
        X4 = triangulate(check_pt_normalized1, check_pt_normalized2, P2_4, True)
        x2_4 = np.dot(P2_4, X4)

        # alternativ: mittels cv2
        # das Ergebnis ist leicht anders als X1, weil openCV pro Punkt eine Gleichung mehr, und somit 3 ansetzt (obwohl lt. Hartley&Zisserman nur 2 von diesen 3 linear unabhängig sind)
        Xcv1 = cv2.triangulatePoints(P1, P2_1, check_pt_normalized1, check_pt_normalized2)
        Xcv1 /= Xcv1[3, :]

        if 0:
            print(X1)
            print(x2_1)
            print(X2)
            print(x2_2)
            print(X3)
            print(x2_3)
            print(X4)
            print(x2_4)
        # okay: nur in einem Fall ist sowohl die Z-Koordinate von X1 positiv (und damit auch die z-Koordinate von x1,
        #                             als auch die z-Koordinate von x2

        # okay: Projektionen der Objektpunkte ergeben wieder die Bildpunkte inkl. Residuen.
        # pt1_check = np.dot(P1,X1)
        # pt2_check = np.dot(P2_1,X1)
        # pt1_check /= pt1_check[2,:] # okay: ~= check_pt_normalized1
        # pt2_check /= pt2_check[2,:] # okay: ~= check_pt_normalized2

        pt1_check = np.dot(P1, X1)
        pt1_check = pt1_check[:2, :] / pt1_check[2, :]
        d1 = check_pt_normalized1 - pt1_check
        pt2_check = np.dot(P2_1, X1)
        pt2_check = pt2_check[:2, :] / pt2_check[2, :]
        d2 = check_pt_normalized2 - pt2_check

        pt1_checkcv = np.dot(P1, Xcv1)
        pt1_checkcv = pt1_checkcv[:2, :] / pt1_checkcv[2, :]
        d1cv = check_pt_normalized1 - pt1_checkcv

        # die L2-Norm der openCV-Lösung ist für alle Punkte kleiner.
        norm2ori = np.sum(d1 ** 2, axis=0) ** .5
        norm2cv = np.sum(d1cv ** 2, axis=0) ** .5

    # suche jene der 4 Lösungen, bei der die (meisten) Objektpunkte (zumindest die inlier) vor beiden Photos liegen.
    # D.h., die z-Koordinaten müssen positiv sein

    # Alternative: cv2.recoverPose
    # n.b.: recoverPose benützt hartcodierte, maximale Distanz==50. (Z-Koordinate) um Punkte,
    #   die sehr weit weg liegen, beim Test, ob R und t einen Objektpunkt vor beiden Kameras ('cheirality check') zu überspringen (für Punkte, die unendlich weit weg liegen, ist nicht definiert, ob sie vor oder hinter der Kamera liegen!)
    # t wird so geschätzt, dass ||t|| == 1. D.h., Punkte, die von Kamera 1 weiter weg liegen als 50 mal die Basislinie, werden verworfen. 
    for i in range(5):
        if i < 2:
            Wl = W
        elif i < 4:
            Wl = W.T
        else:
            raise Exception("Numerical problem?")

        # Achtung!!!
        # Ul ist nur eine Referenz auf die letzte Spalte von U
        # -> Ul *= -1. würde auch U verändern! siehe Ul.flags -> OWNDATA
        # Ul = U[:,-1].reshape((-1,1))
        Ul = U[:, -1].reshape((-1, 1)).copy()
        if i % 2 != 0:
            Ul *= -1.
        R = np.dot(U, np.dot(Wl, V))
        P2 = np.hstack((R, Ul))
        # check_pt1 = pt1[ err < 3**2, : ][0,:].reshape((-1,1))
        # check_pt2 = pt2[ err < 3**2, : ][0,:].reshape((-1,1))
        # check_pt1 = np.vstack((check_pt1,np.ones((check_pt1.shape[1],1))))
        # check_pt2 = np.vstack((check_pt2,np.ones((check_pt2.shape[1],1))))
        # check_pt_normalized1 = np.dot( cameraMatrixInv1, check_pt1 )
        # check_pt_normalized2 = np.dot( cameraMatrixInv2, check_pt2 )
        check_pt_normalized1 = ptNormalized1[:, 0].reshape((-1, 1))
        check_pt_normalized2 = ptNormalized2[:, 0].reshape((-1, 1))

        X = triangulate(check_pt_normalized1, check_pt_normalized2, P2, True)
        Xcv = cv2.triangulatePoints(P1, P2, check_pt_normalized1, check_pt_normalized2)
        Xcv /= Xcv[3, :]
        # X should be close to Xcv

        x2 = np.dot(P2, X)
        # print("{}\n{}\n\n".format(X,x2))

        if X[2] > 0. and x2[2] > 0.:
            t = Ul
            break

    return P2


@contract
def reduction(p1: 'array[Nx2](float)') \
        -> 'array[3x3](float)':
    """reduction matrix: T*p1 -> ~N( 0, sqrt(2) )"""

    ctr = np.mean(p1, axis=0)

    dist = np.sum((p1 - ctr) ** 2., axis=1) ** .5

    meanDist = np.mean(dist)

    scale = 2 ** .5 / meanDist

    T = np.array([[scale, 0, -scale * ctr[0]],
                  [0, scale, -scale * ctr[1]],
                  [0, 0, 1]])
    return T


@contract
def Fmatrix(p1: 'array[Nx2](float)',
            p2: 'array[Nx2](float)',
            enforceRank2: bool = True) \
        -> 'array[3x3](float)':
    # p1,p2: row-wise non-homogeneous points

    T1 = reduction(p1)
    T2 = reduction(p2)

    p1h = np.hstack((p1, np.ones((p1.shape[0], 1)))).T
    p2h = np.hstack((p2, np.ones((p2.shape[0], 1)))).T

    p1r = np.dot(T1, p1h)
    p2r = np.dot(T2, p2h)

    # T1[2,2] == T2[2,2] == 1
    # -> p1r[2,:]==p2r[2,:]==1
    # -> the following is valid:

    A = np.vstack((p2r[0, :] * p1r[0, :], p2r[0, :] * p1r[1, :], p2r[0, :],
                   p2r[1, :] * p1r[0, :], p2r[1, :] * p1r[1, :], p2r[1, :],
                   p1r[0, :], p1r[1, :], np.ones((p1.shape[0]))
                   )).T

    U, S, V = linalg.svd(A)
    F = V[8, :].reshape((3, 3))

    if enforceRank2:
        """enforce rank 2: compute F_1, with det(F_1)=0 and Frobenius-norm |F - F_1| = min"""
        U, S, V = linalg.svd(F)
        S[2] = 0
        F = np.dot(np.dot(U, np.diag(S)), V)

    # Denormalise
    F = np.dot(np.dot(T2.T, F), T1)

    F /= F[2, 2]
    return F


@contract
def algebraicError(p1: 'array[Nx2](float)',
                   p2: 'array[Nx2](float)',
                   F: 'array[3x3](float)') \
        -> float:
    # p1,p2: row-wise non-homogeneous points
    # F: fundamental matrix
    # F is the rank-2 matrix that minimizes the Frobenius norm of F minus F_base, with F_base being the least-squares solution

    F_base = Fmatrix(p1, p2, False)

    # Frobenius norm
    res = np.sum((F - F_base) ** 2)
    return res


@contract
def fitPlaneSVD(XYZ: 'array[Nx3](float)') \
        -> 'array[4](float)':
    """returns plane in Hesse normal form, computed with SVD
    returns [n d]
    where
    x.dot(n)==d
    holds for points x in the plane and
    ||n|| == 1
    """
    rows, cols = XYZ.shape
    # Set up constraint equations of the form  AB = 0,
    # where B is a column vector of the plane coefficients
    # in the form b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
    m = XYZ.mean(axis=0)
    XYZloc = XYZ - m
    [u, d, v] = linalg.svd(XYZloc, 0)
    n = v[-1, :]  # Solution is last column of v.
    assert (abs(linalg.norm(n) - 1.) < 1.e-6)  # should be 1
    if 0:
        p = (np.ones((rows, 1)))
        AB = np.hstack([XYZ, p])
        [u, d, v] = linalg.svd(AB, 0)
        B2 = v[-1, :];  # Solution is last column of v.
        nn = linalg.norm(B2[0:3])
        B2 = B2 / nn
    d = n.dot(m)
    return np.hstack((n, d))
