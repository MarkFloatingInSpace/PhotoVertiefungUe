# -*- coding: cp1252 -*-
import environment
from oriental import config
from oriental import adjust
#adjust.importSubPackagesAndModules()
import oriental.adjust.cost
import oriental.adjust.loss
import oriental.adjust.local_parameterization
from oriental.adjust.cost import abs, sin, cos, acos, atan2

import numpy as np

import unittest

class LineCost( adjust.cost.AutoDiff ):
    def __init__( self, point : 'array[3](float)', normalize : bool = False ):
        self.point = point
        self.normalize = normalize
        super().__init__(3, [6])

    def Evaluate( self, parameters, residuals ):
        line = parameters[0]
        ref, dir = line[:3], line[3:]
        if self.normalize:
            dir = dir / np.sum(dir**2)**.5
            ref = ref - dir * ref.dot(dir)
        assert abs( np.sum(dir**2)**.5 - 1. ) < 1.e-7, 'direction vector does not have unit norm'
        assert ref.dot(dir) < 1.e-7, 'position vector of reference point is not normal to direction vector'
        lambda_ = self.point.dot(dir)
        basePt = ref + lambda_ * dir
        diff = basePt - self.point
        #sumSqr = np.sum(diff**2)
        #if sumSqr == 0.:
        #    return False
        #residuals[0] = sumSqr**.5
        residuals[:] = diff
        return True

class LineParameterization( adjust.local_parameterization._AutoDiff_6_4 ):
    """Local parameterization of a straight line in 3-space, parameterized with 6 parameters in ambient space, and 4 parameters in local space.
    Parametrization in global space:
    - unit direction vector
    - reference point, whose position vector is normal to the direction vector.
    """
    def __init__( self ):
        super().__init__()

    def Evaluate( self, x, delta, xPlusDelta ):
        ref = x[:3]
        dir = x[3:]
        
        assert abs( np.sum(dir**2)**.5 - 1. ) < 1.e-7, 'direction vector does not have unit norm'
        assert ref.dot(dir) < 1.e-7, 'position vector of reference point is not normal to direction vector'
        
        # phi, theta: spherical coordinates of dir, with reversed signs (*-1.)
        phi = - atan2( dir[1], dir[0] )
        # dir has unit norm; thus, we don't need to divide by its norm.
        theta = - acos( dir[2] )
        sp = sin(phi)
        cp = cos(phi)
        st = sin(theta)
        ct = cos(theta)
        # delta is a 2D-point in the local tangent plane of the sphere at the 3D-point dir
        # determine delta's position in the coordinate system of dir by
        # - rotation about y' by theta and subsequent
        # - rotation about z' by phi
        M = np.array([[  cp*ct, sp ],
                      [ -sp*ct, cp ],
                      [     st,  0 ]], dtype=type(dir[0]) )

        dir = dir + M.dot(delta[2:])

        # project dir onto the sphere i.e. normalize its length.
        xPlusDelta[3:] = dir = dir / np.sum(dir**2)**.5

        assert abs( np.sum(dir**2)**.5 - 1. ) < 1.e-7, 'direction vector does not have unit norm'

        ref = ref + M.dot(delta[:2])
        # move ref along dir, such that its position vector becomes normal to dir, again.
        offsetLen = ref.dot(dir)
        xPlusDelta[:3] = ref - dir * offsetLen

        assert xPlusDelta[:3].dot(dir) < 1.e-7, 'position vector of reference point is not normal to direction vector'

        return True

class TestLineParameterization(unittest.TestCase):

    def test_1(self):
        def do_adjust(withLocalParam):
            ref = np.array([2.,1.,.1])
            dir = np.array([.1,.9,.2])
            #dir = np.array([1.,.0,.0])
            dir /= np.sum(dir**2)**.5
            ref -= dir * ref.dot(dir)
            line = np.r_[ ref, dir ]
            self.assertAlmostEqual( line[:3].dot(line[3:]), 0. )
            self.assertAlmostEqual( np.sum(line[3:]**2.)**.5, 1. )
            problem = adjust.Problem()
            loss = adjust.loss.Trivial()
            pts = []
            #for x in range(8):
            #    y = x % 2 - 0.5
            #    pts.append( np.array([x,y,0],float) )
            for x in range(4):
                for y in (1,-1):
                    pts.append( np.array([x,y,0],float) )
            pts = np.array(pts)
            for pt in pts:
                cost = LineCost(pt, not withLocalParam)
                problem.AddResidualBlock( cost, loss, line )
            if withLocalParam:
                parameterization = LineParameterization()
                problem.SetParameterization( line, parameterization )
            options = adjust.Solver.Options()
            options.max_num_iterations = 500
            #options.function_tolerance = 1.e-13
            #options.gradient_tolerance = 1.e-13
            #options.parameter_tolerance = 1.e-13
            summary = adjust.Solver.Summary()
            adjust.Solve( options, problem, summary )

            if withLocalParam:
                local2globalJacobian = parameterization.ComputeJacobian( line )

            self.assertTrue( adjust.isSuccess( summary.termination_type ) )
            cost = np.array([el.cost for el in summary.iterations])
            if 0:
                import matplotlib.pyplot as plt
                plt.plot(pts[:,0],pts[:,1],'.k')
                dir = line[3:]
                if not withLocalParam:
                    dir /= np.sum(dir**2.)**.5
                #plt.plot( [ line[0], line[0] + dir[0]*4 ],
                #          [ line[1], line[1] + dir[1]*4 ],'-r')
                plt.plot( [ 0, 3 ],
                          [ line[1] + -line[0]/dir[0]*dir[1], line[1] + (3-line[0])/dir[0]*dir[1] ],'-r')
                plt.show()
            return cost

        costsWithParam = do_adjust(True)
        costsWithoutParam = do_adjust(False)
        if 0:
            import matplotlib.pyplot as plt
            plt.semilogy( costsWithParam, label='with param')
            plt.semilogy( costsWithoutParam, label='without param')
            plt.axis('tight')
            plt.legend()
            plt.show()
        dummy=1

if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestLineParameterization.test_1',
                       exit=False )
