# -*- coding: cp1252 -*-
"""
statistics
"""

from contracts import contract
import numpy as np
from scipy import linalg, spatial
import scipy.stats

@contract
def relativeDifference( a : float, b : float ) -> float:
    return abs( a - b ) / max(abs(el) for el in (a,b) )

@contract
def geometricMedian( data : 'array[RxC], R>0, C>1', atol=1.e-6, maxIter=200 ) -> 'array[C](float)':
    """Weiszfeld's algorithm

       Returns the L1-mean, i.e. the point with minimal sum of distances to the data points.
       The returned point is thus in general not a data point, not even for odd sample sizes.
       data has R C-dimensional data points
       The L1-mean has a breakdown-point of 50%, just like the median.
       The L1-mean is invariant to Euclidean similarity transformations.
    """
    # note: for the median of a set of rotations, see 2011 Hartley,Aftab,Trumpf - L1 Rotation Averaging using the Weiszfeld Algorithm - conf CVPR

    # Collect identical points, even if this will skew the result if there is more than 1 unique point.
    # Note: np.unique flattens its argument, while we want to compare whole rows!
    # Thus, use a view on rows of unspecified type (binary comparison)
    view = np.ascontiguousarray( data ).view( np.dtype((np.void, data.dtype.itemsize * data.shape[1])) )
    _, idx, counts = np.unique(view, return_index=True, return_counts=True)

    data = data[idx]

    if len(data) <= 1:
        return data[0].astype(float)
    if len(data) <= 2:
        return np.mean( data, axis=0 )

    # The geometric median is equivariant for Euclidean similarity transformations, including translation and rotation
    shift = np.median( data, axis=0 )
    data = data - shift # make a copy
    scale = np.sum( data**2, axis=0 )**.5
    scale = np.mean( scale[scale!=0.] )
    data /= scale
    scaledAtolSqr = (atol/scale)**2

    # with the coordinate-wise mean as initial median estimate, the original Weiszfeld's algorithm does not converge.
    #med = data.mean( axis=0 )
    med = np.zeros_like(shift) # shift set to median!

    # For special cases, compute the result non-iteratively!
    # (E.g. for 3 points that are nearly collinear, the iterative solution would converge really slowly)
    if len(data) == 3:
        sideLens = []
        for iAng in range(3):
            l = (iAng-1) % 3
            r = (iAng+1) % 3
            sideLens.append( linalg.norm(data[l]-data[r]) )
        sideLens = np.array(sideLens)
        if np.linalg.matrix_rank(data) <= 1:
            # if the 3 points are collinear, then med is the point in between the 2 end-points.
            iMaxSide = np.argmax(sideLens)
            med = data[iMaxSide]
        else:
            angs = []
            for iAng in range(2):
                l = (iAng-1) % 3
                c = iAng
                r = (iAng+1) % 3
                d1 = data[l]-data[c]
                d1 /= sideLens[r]
                d2 = data[r]-data[c]
                d2 /= sideLens[l]
                angs.append( np.arccos(d1.dot(d2)) )
            angs.append( np.pi - np.sum(angs) )
            if max(angs) >= 2/3*np.pi: # 120°
                med = data[ np.argmax(angs) ]
            else:
                # Fermat point: the point inside the triangle which subtends an angle of 120° to each three pairs of triangle vertices
                # trilinear coordinates are:
                # x : y : z  = sec(A ? ?/6) : sec(B ? ?/6) : sec(C ? ?/6)
                trilinCoos = 1. / np.cos([ ang - np.pi/6. for ang in angs ])
                weights = sideLens * trilinCoos / sideLens.dot(trilinCoos)
                med = np.sum( ( weights * data.T ).T, axis=0 )
    else:
        rightSingularVectors = None
        if len(data) == 4:
            dataMean = data.mean(axis=0)
            # For 4 coplanar points, if one of the four points is inside the triangle formed by the other three points, then the geometric median is that point.
            # Otherwise, the four points form a convex quadrilateral and the geometric median is the crossing point of the diagonals of the quadrilateral
            if data.shape[1] == 2:
                rightSingularVectors = np.eye(2)
            else:
                # Perform principal component analysis. This is non-robust! However, with 4 points, we have no redundancy left for a robust solution.
                # s : The singular values, sorted in non-increasing order. Of shape (K,), with K = min(M, N)
                U,s,Vh = linalg.svd( data - dataMean, full_matrices=True )
                if s[0]==0. or s[2] / s[0] < 1.e-12:
                    rightSingularVectors = Vh
        if rightSingularVectors is not None:
            dataLoc = ( data - dataMean ).dot( rightSingularVectors.T )
            hull = spatial.ConvexHull( dataLoc[:,:2] )
            if len(hull.vertices) < 3:
                # Transformed points are coincident or collinear (in 2D).
                med = np.mean( data[hull.vertices], axis=0 )
            elif len(hull.vertices) == 3:
                # Chose the original data point that does not form part of the convex hull.
                # That point is either inside the triangle formed by the other points, or it is collinear with its neighbor points (in 2D). In both cases, the point left out is the L1-mean.
                sel = np.ones(4, dtype=np.bool)
                sel[hull.vertices] = False
                med = data[sel][0]
            else:
                # Compute the intersection point of the (transformed) diagonals.
                # For 2-D convex hulls, hull.vertices are in counterclockwise order.
                diff1 = hull.points[hull.vertices[2]] - hull.points[hull.vertices[0]]
                diff2 = hull.points[hull.vertices[3]] - hull.points[hull.vertices[1]]
                lenDiff1 = linalg.norm(diff1)
                lenDiff2 = linalg.norm(diff2)
                n1 = np.r_[ diff1[1], -diff1[0] ] / lenDiff1
                n2 = np.r_[ diff2[1], -diff2[0] ] / lenDiff2
                d1 = n1.dot(hull.points[hull.vertices[0]])
                d2 = n2.dot(hull.points[hull.vertices[1]])
                A = np.vstack(( n1, n2 ))
                pos = linalg.inv(A).dot( np.r_[ d1, d2 ] )
                # interpolate the original points, so we interpolate the other dimensions.
                fraction1 = linalg.norm(pos - hull.points[hull.vertices[0]]) / lenDiff1
                fraction2 = linalg.norm(pos - hull.points[hull.vertices[1]]) / lenDiff2
                pos1 = dataLoc[hull.vertices[0]] + ( dataLoc[hull.vertices[2]] - dataLoc[hull.vertices[0]] ) * fraction1
                pos2 = dataLoc[hull.vertices[1]] + ( dataLoc[hull.vertices[3]] - dataLoc[hull.vertices[1]] ) * fraction2
                med = ( pos1 + pos2 ) / 2.
                # transform back to original CS
                med = med.dot( rightSingularVectors ) + dataMean
        else:
            zeroTol = 1.e-15
            for iIter in range(maxIter):
                medOld = med
                L2norms = np.sum( ( data - med )**2, axis=1 )**.5
                if True:
                    # Vardi, Yehuda; Zhang, Cun-Hui (2000). "The multivariate L1-median and associated data depth". Proceedings of the National Academy of Sciences of the United States of America 97 (4): 1423–1426 (electronic). doi:10.1073/pnas.97.4.1423. MR 1740461
                    # see http://www.statistik.tuwien.ac.at/forschung/CS/CS-2010-4complete.pdf , page 4
                    # implemented in R's package 'robustX'
                    # What they leave undefined is how to compute equality of the current estimate of the median and the data points, using inexact floats!
                    # the R function 'L1median' of package robustX provides the function argument 'zero.tol' for that.
                    notZero = L2norms > zeroTol
                    #notZero = np.max( data - med, axis=1 ) > zeroTol
                    weights = counts[notZero] / L2norms[notZero]
                    T = np.sum( data[notZero].T * weights, axis=1 ) / np.sum( weights )
                    eta = counts[ notZero==False ] # number of data points that are identical to the current median estimate
                    if len(eta) == 0:
                        med = T
                    else:
                        R = np.sum( ( data[notZero] - med ).T * weights, axis=1 )
                        r = linalg.norm( R )
                        med = ( 1. - eta/r ) * T + min( 1., eta/r ) * med
                        #gamma = min( 1, eta / r )
                        #med = ( 1 - gamma ) * T + gamma * med
                        #med = max( 0, 1 - eta / r ) * T + gamma * med
                else:
                    # original Weiszfeld's algorithm
                    notZero = L2norms > zeroTol
                    denoms = np.zeros_like(L2norms)
                    denoms[notZero] = 1. / L2norms[notZero]
                    med = np.sum( data.T * denoms, axis=1 ) / np.sum( denoms )
                #if abs( med.dot(med) - medOld.dot(medOld) ) < medOld.dot(medOld) * 1.e-4:
                # absolute change
                # L2-norm of median's change smaller than atol
                if np.sum( (med - medOld)**2 ) < scaledAtolSqr:
                # max-norm of median's change smaller than atol
                #if np.max( np.abs( med - medOld ) ) < atol**2:
                    break
                # robustX seems to use the relative change of median estimate, L1-norm; threshold taken from default value of robustX
                # This may not break at all, even though the median is converging, if the median converges to zero!
                #if np.abs( med - medOld ).sum() < rtol * np.abs( medOld ).sum():
                #    break
            else:
                raise linalg.LinAlgError("Iterative adjustment of geometric median has not converged within {} iterations".format(maxIter))

    return med * scale + shift

@contract
def circular_mean( alpha : 'array[N](float)' ) -> float:
    '''Computes the mean direction for circular data.
    alpha[rad]
    '''
    return scipy.stats.circmean(alpha)
    #r = np.sum( np.exp( 1j*alpha ) )
    #return np.angle(r)

@contract
def circular_dist( x : 'array[N](float) | float',
                   y : 'array[N](float) | float' ) -> 'array[N](float) | float':
    '''Pairwise difference x_i-y_i around the circle computed efficiently.
    x [rad]
    y [rad]
    '''

    return np.squeeze( np.angle( np.exp( 1j*x ) / np.exp( 1j*y ) ) )

@contract
def circular_dist2( x : 'array[N](float)',
                    y : 'array[M](float)' ) -> 'array[NxM](float)':
    "returns a 2d-array a: a[i,j] is the angular difference: x[i] - y[j]"
    return np.angle(   np.tile( np.atleast_2d( np.exp(1j*x) ).T, (1,len(y)) )
                     / np.tile(                np.exp(1j*y)    , (len(x),1) ) ) # -> (-pi;pi]

@contract
def circular_median( alpha : 'array[N](float),N>0' ) -> float:
    '''Computes the median direction for circular data.
    alpha	sample of angles in radians'''

    alpha = np.remainder( alpha, 2*np.pi )

    dd = circular_dist2( alpha, alpha )
    m1 = np.sum( dd >= 0, axis=0 )
    m2 = np.sum( dd <= 0, axis=0 )

    dm = np.abs( m1 - m2 )
    if len(alpha) % 2 == 1:
        idxs = np.atleast_1d( np.argmin(dm) ) # In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned.
        m = dm[idxs]
    else:
        m = dm.min()
        idxs = np.flatnonzero(dm==m)[:2]

    #if m > 1:
    #    warning('Ties detected.') #ok<WNTAG>

    md = circular_mean( alpha[idxs] )

    meanAlpha = circular_mean(alpha);
    if abs( circular_dist( meanAlpha, md ) ) > abs( circular_dist( meanAlpha, md + np.pi ) ):
        md = np.remainder( md + np.pi, 2*np.pi )
  
    return md

@contract
def minMedMax( arr : 'array[N x ...],N>0') -> 'array[3 x ...]':
    """Efficiently compute minimum, median, and maximum separately for each column.
    If input is 2-dim, returns min/median/max in separate rows.
    The returned array always has the same dtype, even if median is interpolated.
    Thus, for integer arrays, the returned median may be biased towards -infinity."""
    nRows = len(arr)
    if nRows % 2 == 0:
        arr.partition( ( 0, nRows//2-1, nRows//2, -1 ), axis=0 )
        med = ( arr[nRows//2-1] + arr[nRows//2] ) / 2.
    else:
        arr.partition( ( 0, nRows//2, -1 ), axis=0 )
        med = arr[nRows//2]
    if arr.ndim==1:
        return np.array([ arr[0],    # min
                          med,       # median
                          arr[-1] ], # max
                        arr.dtype )
    return np.r_[ '0,2',
                  arr[0],                      # min
                  np.asarray( med, arr.dtype), # median
                  arr[-1] ]                    # max
   
@contract
def minMedMaxWithNorm( arr : 'array[RxC],C>1') -> 'array[3 x C+1]':
    # Instead of computing the norm for every row and extracting the median of it, 
    # let's take the median of squared norms.
    # This works, because squaring does not change sort order.
    # However, if there is an even number of rows, and hence, the median is interpolated, the average would be biased.
    normsSqr = np.sum( arr**2, axis=1 )
    arr = np.c_[ arr, normsSqr ]
    nRows = len(arr)
    if nRows % 2 == 0:
        arr.partition( ( 0, nRows//2-1, nRows//2, -1 ), axis=0 )
        med = np.r_[ arr[nRows//2-1,:-1]     + arr[nRows//2,:-1],
                     arr[nRows//2-1, -1]**.5 + arr[nRows//2, -1]**.5 ]
        med /= 2
    else:
        arr.partition( ( 0, nRows//2, -1 ), axis=0 )
        med = np.r_[ arr[nRows//2,:-1],
                     arr[nRows//2, -1]**.5 ]
    return np.r_[ '0,2',
                  np.r_[ arr[ 0,:-1], arr[ 0,-1]**.5 ], # min
                  np.asarray( med, arr.dtype),          # median
                  np.r_[ arr[-1,:-1], arr[-1,-1]**.5 ] ]# max
