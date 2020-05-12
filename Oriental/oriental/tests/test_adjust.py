# -*- coding: cp1252 -*-
import environment
import oriental
from oriental import config, adjust, log, ori
import oriental.adjust.cost
import oriental.adjust.loss
import oriental.adjust.local_parameterization
from oriental.adjust import parameters

adjust.setCeresMinLogLevel( log.Severity.info )

import numpy as np
from scipy import linalg

from contracts import ContractException

import unittest
from collections import namedtuple

class TestAdjustTorlegard(unittest.TestCase):

    def setUp( self ):
        self.angles = parameters.EulerAngles( parametrization=adjust.EulerAngles.omfika, array=np.array([.1,-.02,.3]) )
        self.prjCtr = np.array([ 0., 0., 10. ])
        self.ior    = np.array([ 150., -150., 100. ])
        self.adp    = parameters.ADP( normalizationRadius=75. )

        self.points = np.array([ [  10.,  10., 0. ],
                                 [ -10.,  10., 0. ],
                                 [ -10., -10., 0. ],
                                 [  10., -10., 0. ] ] )

        self.obs = np.array([ [ self.ior[0] +100.+.432, self.ior[1] +100.+.472 ],
                              [ self.ior[0] -100.-.654, self.ior[1] +100.+.581 ],
                              [ self.ior[0] -100.+.589, self.ior[1] -100.+.240 ],
                              [ self.ior[0] +100.-.487, self.ior[1] -100.+.541 ] ] )

        self.loss = adjust.loss.Trivial()

        self.problem = adjust.Problem()

        self.residualIDs = []

        for idx in range( self.points.shape[0] ):
            # 1 residual block per image point
            self.residualIDs.append(
                self.problem.AddResidualBlock(
                    adjust.cost.PhotoTorlegard( self.obs[idx][0], self.obs[idx][1] ),
                    self.loss,
                    self.prjCtr,
                    self.angles,
                    self.ior,
                    self.adp,
                    self.points[idx] )
            )

            self.problem.SetParameterBlockConstant( self.points[idx] )

        self.problem.SetParameterBlockConstant( self.ior )
        self.problem.SetParameterBlockConstant( self.adp )

    def test_cholmodDiagQxx(self):
        log.setScreenMinSeverity( log.Severity.verbose )
        logger = log.Logger("test_cholmodDiagQxx")
        if 0: # this takes really long
            import pickle
            with open( r'D:\_data\Forkert_Oblique-Mounting\adjust_all\OrientAL\adp_1,2,3,4,5,6\jacobian.pickle', 'rb') as fin:
                jacobian = pickle.load( fin )
            diagQxx = adjust.diagQxx( jacobian )

        #from scipy.io import loadmat
        #import pickle
        #jacobian = loadmat( r'D:\_data\Forkert_Oblique-Mounting\adjust_all\OrientAL\adp_2,3,4,5\old\jacobian.mat' )['jacobian']
        #diagQxx = adjust.diagQxx( jacobian )
        #sigmas = 1.0382 * diagQxx**.5
        #pickleFn = r'D:\_data\Forkert_Oblique-Mounting\adjust_all\OrientAL\adp_2,3,4,5\old\sparseQxx.pickle'
        #try:
        #    with open( pickleFn, 'rb' ) as fin:
        #        sparseQxx = pickle.load( fin )
        #except OSError:
        #    sparseQxx = adjust.sparseQxx( jacobian )
        #    with open( pickleFn, 'wb' ) as fout:
        #        pickle.dump( sparseQxx, fout, protocol=pickle.HIGHEST_PROTOCOL )
        #sparseQxx.check_format()

        # sparseQxx is too large to be visualized with plt.spy
        # however, sub-blocks can be visualized!
        # import oriental.utils.pyplot as plt
        # plt.spy( sparseQxx[-100:,-100:] )

        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.set_parameter_blocks( [ self.prjCtr, self.angles ] )
        jacobian, = self.problem.Evaluate( evalOpts, residuals=False,jacobian=True)
        assert jacobian.shape[1] == np.unique( jacobian.nonzero()[1] ).size, "zero columns in jacobian"
        jacobianDense = jacobian.toarray()

        Qxx = linalg.inv( jacobianDense.T.dot( jacobianDense ) )
        #logger.info("Calling diagQxx")
        diagQxx = adjust.diagQxx( jacobian )
        #logger.info("diagQxx returned")
        np.testing.assert_almost_equal( Qxx.diagonal(), diagQxx )

        # remove zero columns (constant parameter blocks)
        #jacobian = jacobian[:,np.unique(jacobian.nonzero()[1])]
        #jacobian.sort_indices()
        #diagQxx = adjust.cholmodDiagQxx( jacobian )
        #diagQxx = adjust.cholmodDiagQxxSuperNodal( jacobian )
        sparseQxx = adjust.sparseQxx( jacobian )
        sparseQxx.check_format()
        nz = sparseQxx.nonzero()
        np.testing.assert_array_almost_equal( sparseQxx[nz].flat, Qxx[nz] )

        #dummy=1
        #import oriental.utils.pyplot as plt
        #plt.spy( sparseQxx )
        #logger.info("Calling cholmodDiagQxxSuperNodal")
        #diagQxx = adjust.cholmodDiagQxxSuperNodal( jacobian )
        #print(diagQxx)
        #logger.info("cholmodDiagQxxSuperNodal returned")
        #logger.info("Calling cholmodDiagQxxSimplicial")
        #ret = adjust.cholmodDiagQxxSimplicial( jacobian )
        #logger.info("cholmodDiagQxxSimplicial returned")
        #diagQxx = adjust.cholmodDiagQxx( jacobian )
        dummy=1

    def test_redundancyComponents(self):
        log.setScreenMinSeverity( log.Severity.verbose )
        logger = log.Logger("test_cholmodDiagQxx")

        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.set_parameter_blocks( [ self.prjCtr, self.angles ] )
        jacobian = -self.problem.Evaluate( evalOpts, residuals=False,jacobian=True)[0]
        assert jacobian.shape[1] == np.unique( jacobian.nonzero()[1] ).size, "zero columns in jacobian"
        jacobianDense = jacobian.toarray()

        nObs, nUkn = jacobian.shape
        N = (jacobian.T @ jacobian).toarray()
        C = linalg.cholesky(N, lower=False)
        Qxx = linalg.cho_solve((C, False), np.eye(nUkn))
        QvvP = -(jacobian @ Qxx @ jacobian.T - np.eye(nObs))
        redundancyComponentsFull = np.diag(QvvP)

        redundancyComponents = adjust.redundancyComponents(jacobian)
        np.testing.assert_almost_equal( redundancyComponentsFull, redundancyComponents )

    def test_evalOptions(self):
        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.set_parameter_blocks( [ self.prjCtr ] )
        evalOpts.set_residual_blocks( [ self.residualIDs[0] ] )
        jacobian, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        jacobian2, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        np.testing.assert_allclose( jacobian.todense(), jacobian2.todense() )


        residuals_apriori, = self.problem.Evaluate()
        residuals_apriori2, = self.problem.Evaluate()
        np.testing.assert_allclose( residuals_apriori, residuals_apriori2 )

        jacobian, = self.problem.Evaluate( residuals=False, jacobian=True )
        jacobian2, = self.problem.Evaluate( residuals=False, jacobian=True )
        np.testing.assert_allclose( jacobian.todense(), jacobian2.todense() )

        evalOpts = adjust.Problem.EvaluateOptions()

        evalOpts.set_residual_blocks( [ self.residualIDs[0] ] )
        assert len(evalOpts.get_residual_blocks() ) == 1
        # the following test does not succeed, because even though these opaque pointers refer to the same address,
        # they are distinct Python objects. As no special comparison method is defined for them, they are compared by their own memory addresses
        #self.assertEqual( evalOpts.get_residual_blocks()[0], self.residualIDs[0] )

        residuals_apriori, = self.problem.Evaluate( evalOpts )
        self.assertTrue( residuals_apriori.shape == (2,), "We asked for the residuals of the first residual block, which is expected to return 2 residuals" )

        residuals_apriori2, = self.problem.Evaluate( evalOpts )
        np.testing.assert_allclose( residuals_apriori, residuals_apriori2 )

        jacobian, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        self.assertTrue( jacobian.shape == (2,30), "We asked for the jacobian of the first residual block and all parameter blocks, which is expected to be a 2x30 matrix" )
        jacobian2, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        np.testing.assert_allclose( jacobian.todense(), jacobian2.todense() )

        evalOpts.set_parameter_blocks( [ self.angles ] )
        jacobian, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        self.assertTrue( jacobian.shape == (2,3), "We asked for the jacobian of the first residual block and the parameter block 'angles', which is expected to be a 2x3 matrix" )

        jacobian2, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        np.testing.assert_allclose( jacobian.todense(), jacobian2.todense() )

        evalOpts.set_parameter_blocks( [ self.angles ] )
        jacobian3, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        np.testing.assert_allclose( jacobian.todense(), jacobian3.todense() )

        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.set_parameter_blocks( [ self.prjCtr, self.angles ] )
        j1, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        #self.assertTrue( jacobian.shape == (2,3), "We asked for the jacobian of all residual blocks and the parameter block 'prjCtr', which is expected to be a 8x3 matrix" )
        j2, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        # the following fails often, but not always:
        # Probable reason has been fixed in trunc on Tue Dec 03 09:28:14 2013 -0800:
        # https://ceres-solver.googlesource.com/ceres-solver/+/324eccb5f6ce2a1a0061ec9f3c40778a029a2d97
        #Calling Problem::Evaluate mutates the state of the parameter blocks.
        #In particular, depending on the set and order of parameter blocks
        #passed to the evaluate call, it will change the internal indexing
        #used by the Program object used by ProblemImpl. This needs to be
        #undone before Evaluate returns, otherwise the Problem object
        #is in an invalid state.

        # wk: note that Ceres 1.8.0 was released before that, on Nov. 12
        # wk: copied relevant parts of bug-fix into ceres-1.8.0
        np.testing.assert_allclose( j1.todense(), j2.todense() )

        cost, residuals, gradient, jacobi = self.problem.Evaluate( evalOpts, cost=True, residuals=True, gradient=True, jacobian=True )

        np.testing.assert_allclose( gradient, jacobi.T.dot( residuals )  )

        # cost is half the residual square sum
        np.testing.assert_allclose( cost, residuals.dot(residuals) / 2.  )

    def test_covariance(self):
        options = adjust.Solver.Options()
        options.function_tolerance = 1.e-14
        options.gradient_tolerance = options.function_tolerance * 1.e-4
        summary = adjust.Solver.Summary()

        adjust.Solve( options, self.problem, summary )

        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.set_parameter_blocks( [ self.prjCtr, self.angles ] )
        jacobi, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        QxxJacobi = linalg.inv( jacobi.T.dot(jacobi).todense() )

        QxxMatlab = np.array([
            [ 1.2485e-002, -4.3359e-007,  9.7216e-006,  1.8259e-008,  4.9930e-004,  5.0881e-006],
            [-4.3359e-007,  1.2483e-002, -3.2012e-005, -4.9927e-004, -1.6650e-008,  2.4020e-007],
            [ 9.7216e-006, -3.2012e-005,  1.2484e-003,  3.0299e-007,  2.7439e-007,  2.6501e-009],
            [ 1.8259e-008, -4.9927e-004,  3.0299e-007,  2.4966e-005,  6.2678e-010, -3.7164e-009],
            [ 4.9930e-004, -1.6650e-008,  2.7439e-007,  6.2678e-010,  2.4964e-005,  1.9204e-007],
            [ 5.0881e-006,  2.4020e-007,  2.6501e-009, -3.7164e-009,  1.9204e-007,  1.2493e-005]
        ])
        # Achtung: adjPho.m rechnet Winkel in Radiant
        QxxMatlabGon = QxxMatlab.copy()
        QxxMatlabGon[:3,3:] *= 200./np.pi
        QxxMatlabGon[3:,:3] *= 200./np.pi
        QxxMatlabGon[3:,3:] *= (200./np.pi)**2

        covariance = adjust.Covariance()
        covariance.Compute( [ ( self.prjCtr, self.prjCtr ),
                              ( self.prjCtr, self.angles ),
                              ( self.angles, self.angles ) ],
                            self.problem )
        QxxOrientAL = np.zeros( (6,6) )

        covPrjCtr = covariance.GetCovarianceBlock( self.prjCtr, self.prjCtr )
        self.assertEqual( covPrjCtr.shape, (3,3) )
        QxxOrientAL[:3,:3] = covPrjCtr
        np.testing.assert_allclose( covPrjCtr, QxxMatlabGon[:3,:3], rtol=1.e-4 )

        covPrjCtrAngles = covariance.GetCovarianceBlock( self.prjCtr, self.angles )
        self.assertEqual( covPrjCtrAngles.shape, (3,3) )
        QxxOrientAL[:3,3:] = covPrjCtrAngles
        np.testing.assert_allclose( covPrjCtrAngles, QxxMatlabGon[:3,3:], rtol=1.e-4 )

        covAnglesAngles = covariance.GetCovarianceBlock( self.angles, self.angles )
        self.assertEqual( covAnglesAngles.shape, (3,3) )
        QxxOrientAL[3:,3:] = covAnglesAngles
        np.testing.assert_allclose( covAnglesAngles, QxxMatlabGon[3:,3:], rtol=1.e-4 )
        QxxOrientAL = np.triu( QxxOrientAL, 1 ) + np.diag( np.diag( QxxOrientAL ) ) + np.triu( QxxOrientAL, 1 ).T
        np.testing.assert_allclose( QxxOrientAL, QxxJacobi )
        np.testing.assert_allclose( QxxOrientAL, QxxMatlabGon, rtol=1.e-4 )

        with self.assertRaises(oriental.Exception):
            covariance.Compute( [ ( 0, self.prjCtr ) ], self.problem )

    def test_solve(self):
        angles_apri = self.angles.copy()
        prjCtr_apri = self.prjCtr.copy()

        residuals_apri, = self.problem.Evaluate()

        options = adjust.Solver.Options()
        summary = adjust.Solver.Summary()

        adjust.Solve( options, self.problem, summary )

        self.assertTrue( adjust.isSuccess( summary.termination_type ), "adjustment was unsuccessful" )
        self.assertTrue( summary.final_cost < summary.initial_cost, "adjustment has not reduced the cost" )

        residuals_apost, = self.problem.Evaluate();

        #print( summary.BriefReport() )
        #print( "prjCtr: {} -> {}".format( prjCtr_apri, self.prjCtr ) )
        #print( "angles: {} -> {}".format( np.array(angles_apri), np.array(self.angles) ) )
        #
        #print("residuals:")
        #for idx in range( 0, residuals_apri.shape[0], 2 ):
        #    print( "{} -> {}".format( residuals_apri[idx:idx+2], residuals_apost[idx:idx+2] ) )

    def test_solve_weighted(self):
        stdDevs = np.array([ 2., 3. ])
        Pblock = linalg.inv( np.diag( stdDevs**2 ) )
        # sqrt(P) for each residual block
        sqrtPBlock = linalg.sqrtm( Pblock ).copy() # copy -> make C-contiguous, own data

        self.problem = adjust.Problem()
        self.loss = adjust.loss.Trivial()

        self.residualIDs = []

        for idx in range( self.points.shape[0] ):
            # 1 residual block per image point
            self.residualIDs.append(
                self.problem.AddResidualBlock(
                    adjust.cost.PhotoTorlegard( self.obs[idx][0], self.obs[idx][1], sqrtPBlock ),
                    self.loss,
                    self.prjCtr,
                    self.angles,
                    self.ior,
                    self.adp,
                    self.points[idx] )
            )

            self.problem.SetParameterBlockConstant( self.points[idx] )

        self.problem.SetParameterBlockConstant( self.ior )
        self.problem.SetParameterBlockConstant( self.adp )

        angles_apri = self.angles.copy()
        prjCtr_apri = self.prjCtr.copy()

        residuals_apri, = self.problem.Evaluate();

        options = adjust.Solver.Options()
        options.function_tolerance = 1.e-14
        options.gradient_tolerance = options.function_tolerance * 1.e-4

        summary = adjust.Solver.Summary()

        adjust.Solve( options, self.problem, summary )

        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.set_parameter_blocks( [ self.prjCtr, self.angles ] )
        jacobi, = self.problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        QxxJacobi = linalg.inv( jacobi.T.dot(jacobi).todense() )

        # this works: altering an element of the sqrtP-Matrix affects all costs that were constructed with that numpy.ndarray
        #sqrtPBlock[0,0] = 0.8
        #adjust.Solve( options, self.problem, summary )

        self.assertTrue( adjust.isSuccess( summary.termination_type ), "adjustment was unsuccessful" )
        self.assertTrue( summary.final_cost < summary.initial_cost, "adjustment has not reduced the cost" )

        # Achtung: residuals_apost sind mit sqrtP multipliziert.
        residuals_apost, = self.problem.Evaluate();

        #print( summary.BriefReport() )
        #print( "prjCtr: {} -> {}".format( prjCtr_apri, self.prjCtr ) )
        #print( "angles: {} -> {}".format( np.array(angles_apri), np.array(self.angles) ) )
        #
        #print("residuals:")
        #for idx in range( 0, residuals_apri.shape[0], 2 ):
        #    print( "{} -> {}".format( residuals_apri[idx:idx+2], residuals_apost[idx:idx+2] ) )


        QxxMatlab = np.array([
            [ 9.9914e-002,  4.1253e-005,  1.1087e-004, -1.6831e-006,  4.4958e-003,  2.8194e-005 ],
            [ 4.1253e-005,  6.2451e-002, -1.2867e-004, -1.9983e-003,  1.6288e-006,  2.1230e-006 ],
            [ 1.1087e-004, -1.2867e-004,  6.9179e-003, -3.0935e-006,  4.5699e-006,  1.9992e-008 ],
            [-1.6831e-006, -1.9983e-003, -3.0935e-006,  9.9924e-005, -8.4790e-008, -6.0176e-008 ],
            [ 4.4958e-003,  1.6288e-006,  4.5699e-006, -8.4790e-008,  2.2478e-004,  1.0641e-006 ],
            [ 2.8194e-005,  2.1230e-006,  1.9992e-008, -6.0176e-008,  1.0641e-006,  6.9209e-005 ]
        ])
        QxxMatlabGon = QxxMatlab.copy()
        QxxMatlabGon[:3,3:] *= 200./np.pi
        QxxMatlabGon[3:,:3] *= 200./np.pi
        QxxMatlabGon[3:,3:] *= (200./np.pi)**2
        sigma0_Matlab = 2.9188e-002

        covariance = adjust.Covariance()
        covariance.Compute(
            [ ( self.prjCtr, self.prjCtr ),
              ( self.prjCtr, self.angles ),
              ( self.angles, self.angles ) ],
            self.problem )
        Qxx = np.zeros( (6,6) )

        covPrjCtr = covariance.GetCovarianceBlock( self.prjCtr, self.prjCtr )
        self.assertEqual( covPrjCtr.shape, (3,3) )
        Qxx[:3,:3] = covPrjCtr
        np.testing.assert_allclose( covPrjCtr, QxxMatlabGon[:3,:3], rtol=1.e-4 )

        covPrjCtrAngles = covariance.GetCovarianceBlock( self.prjCtr, self.angles )
        self.assertEqual( covPrjCtrAngles.shape, (3,3) )
        Qxx[:3,3:] = covPrjCtrAngles
        np.testing.assert_allclose( covPrjCtrAngles, QxxMatlabGon[:3,3:], rtol=1.e-4 )

        covAnglesAngles = covariance.GetCovarianceBlock( self.angles, self.angles )
        self.assertEqual( covAnglesAngles.shape, (3,3) )
        Qxx[3:,3:] = covAnglesAngles
        np.testing.assert_allclose( covAnglesAngles, QxxMatlabGon[3:,3:], rtol=1.e-4 )

        Qxx = np.triu( Qxx, 1 ) + np.diag( np.diag( Qxx ) ) + np.triu( Qxx, 1 ).T
        np.testing.assert_allclose( Qxx, QxxJacobi )
        np.testing.assert_allclose( Qxx, QxxMatlabGon, rtol=1.e-4 )

        cost_apost, = self.problem.Evaluate( cost=True, residuals=False );
        sigma0 = ( cost_apost*2 / ( 8 - 6 ) )**.5
        np.testing.assert_allclose( sigma0, sigma0_Matlab, rtol=1.e-4 )

        Qll = np.diag( np.tile( stdDevs**2, 4 ) )
        #Qll = sparse.diags( np.tile( stdDevs**2, 4 ), offsets=0 )
        # Achtung: jacobi ist bereits multipliziert mit sqrtP !
        AsqrtP = jacobi.toarray()
        # dividiere jede Zeile durch sqrtP == multipliziere jede Zeile mit stdDev
        A = (AsqrtP.T * np.tile( stdDevs, 4 ) ).T
        #P = linalg.inv( Qll )
        P = np.diag( np.tile( 1. / stdDevs**2, 4 ) )
        np.testing.assert_allclose( np.trace( P.dot( Qll ) ), 8. ) # n = 8

        Qvv = Qll - A.dot( Qxx ).dot( A.T )
        redundanzAnteile = ( P.dot( Qvv ) ).diagonal()
        np.testing.assert_allclose( redundanzAnteile.sum(), 8. - 6. ) # r = 8 - 6

    def test_parameterOrdering(self):
        options = adjust.Solver.Options()
        options.linear_solver_ordering = adjust.ParameterBlockOrdering()    

        for pt in self.points:
            options.linear_solver_ordering.AddElementToGroup( pt, 0 )

        options.linear_solver_ordering.AddElementToGroup( self.angles, 1 )
        options.linear_solver_ordering.AddElementToGroup( self.prjCtr, 1 )
        options.linear_solver_ordering.AddElementToGroup( self.ior, 1 )
        options.linear_solver_ordering.AddElementToGroup( self.adp, 1 )

        lenBefore = options.linear_solver_ordering.GroupSize(1)
        summary = adjust.Solver.Summary()
        adjust.Solve( options, self.problem, summary )
        lenAfter = options.linear_solver_ordering.GroupSize(1)
        # ceres removes constant parameter blocks from linear_solver_ordering.
        # Make sure that our cloning works.
        self.assertEqual( lenBefore, lenAfter )

        # ensure that using the same options multiple times does not throw.
        adjust.Solve( options, self.problem, summary )

    def test_incompleteParameterOrdering(self):
        '''Ceres seems to have changed behaviour.
        First, parameter blocks not found in adjust.Solver.Options.ParameterBlockOrdering were simply considered to belong to group 0.
        Now, all parameters are required to be part of ParameterBlockOrdering. However, it does not throw. Maybe because the block is too small, so the ordering is not used at all?'''
        options = adjust.Solver.Options()
        options.linear_solver_ordering = adjust.ParameterBlockOrdering()    

        for pt in self.points:
            options.linear_solver_ordering.AddElementToGroup( pt, 0 )

        options.linear_solver_ordering.AddElementToGroup( self.angles, 1 )
        #options.linear_solver_ordering.AddElementToGroup( self.prjCtr, 1 )
        options.linear_solver_ordering.AddElementToGroup( self.ior, 1 )
        options.linear_solver_ordering.AddElementToGroup( self.adp, 1 )

        summary = adjust.Solver.Summary()
        # ensure that this throws, but doesn't crash
        #self.assertRaises( oriental.Exception, adjust.Solve, options, self.problem, summary )
        # ensure that it doesn't crash, even if it doesn't throw
        adjust.Solve( options, self.problem, summary )


class TestRelative(unittest.TestCase):
    def setUp( self ):
        self.objPts = np.array([[  1.,  1., -1. ],
                                [ -1.,  1., -1. ],
                                [  0.,  0., -1. ],
                                [ -1., -1., -1. ],
                                [  1., -1., -1. ]])

        self.angles = [ parameters.EulerAngles( array=np.array([ 0.,   0., 0.]) ),
                        parameters.EulerAngles( array=np.array([ 0.,  20., 0.]) )  ]
        self.prcs = [ np.array([ 0., 0., 0. ]),
                      np.array([ 1., 0., 0. ]) ]
        self.ior = np.array([ 100., -100., 100. ])
        self.adp = parameters.ADP( normalizationRadius=100. )

    def setupProblem( self ):
        self.imgObss = [ ori.projection( self.objPts, prc, angle, self.ior, self.adp ) + np.random.randn( self.objPts.shape[0], 2 ) for prc,angle in zip(self.prcs,self.angles) ]

        self.problem = adjust.Problem()
        self.loss = adjust.loss.Trivial()
        for imgObs,prc,angle in zip(self.imgObss,self.prcs,self.angles):
            for imgPt,objPt in zip(imgObs,self.objPts):
                cost = adjust.cost.PhotoTorlegard( *imgPt )
                self.problem.AddResidualBlock(
                    cost,
                    self.loss,
                    prc,
                    angle,
                    self.ior,
                    self.adp,
                    objPt )

        self.problem.SetParameterBlockConstant( self.prcs[0] )
        self.problem.SetParameterBlockConstant( self.angles[0] )
        self.problem.SetParameterBlockConstant( self.adp )
        # UnitNorm3 does not reduce the number of unknowns! Thus, covariance estimation (not Solve!) with UnitNorm3 fails due to a rank deficit (N-matrix not positive definite).
        #parameterization = adjust.local_parameterization.UnitNorm3()
        parameterization = adjust.local_parameterization.UnitSphere()
        self.problem.SetParameterization( self.prcs[1], parameterization )

    def solveAndComputeCovariance( self ):
        options = adjust.Solver.Options()
        summary = adjust.Solver.Summary()
        adjust.Solve( options, self.problem, summary )

        self.assertTrue( adjust.isSuccess( summary.termination_type ) )

        redundancy = summary.num_residuals_reduced - summary.num_effective_parameters_reduced
        self.assertGreaterEqual( redundancy, 0 )
        sigma0 = ( summary.final_cost * 2 / redundancy ) **.5 if redundancy > 0 else 0.

        covOpts = adjust.Covariance.Options()
        covOpts.apply_loss_function = False
        covOpts.algorithm_type = adjust.CovarianceAlgorithmType.DENSE_SVD
        covariance = adjust.Covariance( covOpts )
        paramBlockPairs = [ (par,par) for par in self.prcs + self.angles + [self.ior] + [self.adp] + [objPt for objPt in self.objPts ] ]
        covariance.Compute( paramBlockPairs, self.problem )
        #print( 'cofactors:' if sigma0==0. else 'std.devs.:' )
        for paramBlockPair in paramBlockPairs:
            cofactorBlock = covariance.GetCovarianceBlock( *paramBlockPair )
            vals = np.diag(cofactorBlock)**0.5
            if sigma0 != 0.:
                vals *= sigma0
            #print( vals )

    def test_iorConst(self):
        self.setupProblem()
        self.problem.SetParameterBlockConstant( self.ior )
        self.solveAndComputeCovariance()

    def test_iorVariable(self):
        self.angles.append( parameters.EulerAngles( array=np.array([ -20., 0., 0.]) ) )
        self.prcs.append( np.array([ 0., 1., 0. ]) )
        self.objPts = np.vstack(( self.objPts,
                                  np.array([  .5,  .5, -2. ]) ))
        self.setupProblem()
        self.solveAndComputeCovariance()

    def test_iorVariable2(self):
        self.angles.append( parameters.EulerAngles( array=np.array([ -20., 0., 0.]) ) )
        self.prcs.append( np.array([ 0., 1., 0. ]) )
        self.objPts = np.vstack(( self.objPts,
                                  np.array([  .5,  .5, -2. ]) ))
        self.setupProblem()
        self.problem.SetParameterBlockConstant( self.ior )
        self.solveAndComputeCovariance()

        self.problem.SetParameterBlockVariable( self.ior )

        covOpts = adjust.Covariance.Options()
        covOpts.apply_loss_function = False
        covOpts.algorithm_type = adjust.CovarianceAlgorithmType.DENSE_SVD
        covariance = adjust.Covariance( covOpts )
        paramBlockPairs = [ (par,par) for par in self.prcs + self.angles + [self.ior] + [self.adp] + [objPt for objPt in self.objPts ] ]
        covariance.Compute( paramBlockPairs, self.problem )
        #print( 'cofactors:' )
        for paramBlockPair in paramBlockPairs:
            cofactorBlock = covariance.GetCovarianceBlock( *paramBlockPair )
            vals = np.diag(cofactorBlock)**0.5
            #print( vals )

class CustomCost( adjust.cost.AutoDiff ):
    """Cost function that evaluates `10-x`"""

    def __init__( self, alterParameters=False, alterParameterElement=False, assignResiduals=True ):
        """Initialize an instance of CustomCost"""

        numResiduals = 1
        parameterSizes = (1,)
        # initialize the base class
        super().__init__( numResiduals, parameterSizes )

        self.alterParameters = alterParameters
        self.alterParameterElement = alterParameterElement
        self.assignResiduals = assignResiduals

    def Evaluate( self, parameters, residuals ):
        """Evaluate the ``residuals`` and possibly, their derivatives w.r.t. ``parameters``"""

        # Store the data type of the ``parameters``.
        # It is identical to the data type of ``residuals``
        dt = type(parameters[0][0])

        if self.alterParameters:
            parameters[0][0] = 2.

        if self.alterParameterElement:
            elem = parameters[0][0]
            elem += dt(3.)

        # Compute the residual and possibly, also its derivatives
        # Use ``dt(10.)`` to ensure that the data type (possibly ``Jet``) assigned to ``residuals`` remains unchanged
        if self.assignResiduals:
            residuals[0] = dt(10.) - parameters[0][0]

        # indicate success
        return True

class TestCustomCostSimple(unittest.TestCase):

    def setUp( self ):
        self.loss = adjust.loss.Trivial()
        self.problem = adjust.Problem()
        self.x = np.array([ 5.0 ])
        self.options = adjust.Solver.Options()
        self.summary = adjust.Solver.Summary()

    def test_ok(self):
        cost = CustomCost()
        self.problem.AddResidualBlock( cost, self.loss, self.x )
        adjust.Solve( self.options, self.problem, self.summary )

        self.assertTrue( adjust.isSuccess( self.summary.termination_type ), "adjustment was unsuccessful" )
        self.assertTrue( self.summary.final_cost < self.summary.initial_cost, "adjustment has not reduced the cost" )

    def test_residualsNotAssigned(self):
        cost = CustomCost(assignResiduals=False)
        self.problem.AddResidualBlock( cost, self.loss, self.x )

        self.assertRaises( oriental.Exception, self.problem.Evaluate )
        self.assertRaises( oriental.Exception, adjust.Solve, self.options, self.problem, self.summary )

    def test_alterParameters(self):
        cost = CustomCost(alterParameters=True)
        self.problem.AddResidualBlock( cost, self.loss, self.x )

        self.assertRaises( oriental.ExcPython, self.problem.Evaluate )
        self.assertRaises( oriental.ExcPython, adjust.Solve, self.options, self.problem, self.summary )

    def test_alterParameterElement(self):
        x_orig = self.x.copy()
        cost = CustomCost(alterParameterElement=True)
        self.problem.AddResidualBlock( cost, self.loss, self.x )

        np.testing.assert_array_equal( x_orig, self.x )

        # An exception is raised only if cost.pyd has been compiled with ORIENTAL_ADJUST_COST_AUTODIFF_CHECK_PARAMS
        #self.assertRaises( oriental.Exception, self.problem.Evaluate, residuals=False, jacobian=True )

class CustomCostPho( adjust.cost.AutoDiff ):
    """adjust.cost.PhotoTorlegard nachgebaut in Python, für ADP=0"""

    def __init__( self, obs_x, obs_y, rotMatParametrization=adjust.EulerAngles.omfika ):
        """Initialize an instance of CustomTorlegardCost"""
        self.obs_x = obs_x
        self.obs_y = obs_y
        self.rotMatParametrization = rotMatParametrization
        numResiduals = 2
        parameterSizes = (3,3,3,3) # PRC, ROT, IOR, PT
        # initialize the base class
        super().__init__( numResiduals, parameterSizes )

    def Evaluate( self, parameters, residuals ):
        """Evaluate the ``residuals`` and possibly, their derivatives w.r.t. ``parameters``"""

        # Store the data type of the ``parameters``.
        # It is identical to the data type of ``residuals``
        dt = type(parameters[0][0])

        prjCtr = parameters[0]
        angles = parameters[1]
        ior    = parameters[2]
        objPt  = parameters[3]

        gon_to_radians = np.pi / 200.0
        from oriental.adjust.cost import sin, cos

        if self.rotMatParametrization == adjust.EulerAngles.omfika:
            om = angles[0] * gon_to_radians
            fi = angles[1] * gon_to_radians
            ka = angles[2] * gon_to_radians

            # functions in module 'adjust.cost' support both float and adjust.cost.Jet as arguments,
            # unlike those in module 'math'!
            cOm = cos(om)
            sOm = sin(om)
            cFi = cos(fi)
            sFi = sin(fi)
            cKa = cos(ka)
            sKa = sin(ka)

            # create an uninitialized 3x3-array of the same data type as parameters and residuals
            R = np.empty( (3,3), dtype=dt )

            R[0][0] = cFi*cKa;               R[0][1] = -cFi*sKa;              R[0][2] = sFi;

            R[1][0] = cOm*sKa + sOm*sFi*cKa; R[1][1] = cOm*cKa - sOm*sFi*sKa; R[1][2] = -sOm*cFi;

            R[2][0] = sOm*sKa - cOm*sFi*cKa; R[2][1] = sOm*cKa + cOm*sFi*sKa; R[2][2] = cOm*cFi;

            if dt != adjust.cost.Jet:
                # Doesn't work with Jet, because assert_allclose indirectly calls the numpy-ufunc `isinf`, which cannot operate on Jet.
                np.testing.assert_allclose( R, adjust.cost.omFiKaToRotationMatrix( angles ) )

        elif self.rotMatParametrization == adjust.EulerAngles.alzeka:
            alpha = angles[0] * gon_to_radians
            zeta  = angles[1] * gon_to_radians
            kappa = angles[2] * gon_to_radians

            # Stelle aus diesen die Rotationsmatrix zusammen.
            R_alpha = np.array([ [ cos(alpha), -sin(alpha),  0. ],
                                 [ sin(alpha),  cos(alpha),  0. ],
                                 [      0    ,       0    ,  1. ] ],
                                 dtype=dt )

            R_zeta  = np.array([ [  cos(zeta),   0,  sin(zeta) ],
                                 [      0    ,   1,     0      ],
                                 [ -sin(zeta),   0,  cos(zeta) ] ],
                                 dtype=dt )

            R_kappa = np.array([ [ cos(kappa), -sin(kappa),  0. ],
                                 [ sin(kappa),  cos(kappa),  0. ],
                                 [      0.   ,       0.   ,  1. ] ],
                                 dtype=dt )

            # Beachte: das Objekt- wird ins Kamerakoordinatensystem mit der Transponierten der Rotationsmatrix gedreht.
            # Deshalb ist die Reihenfolge der Rotationen hier umgekehrt:
            # zuerst um kappa, dann zeta, dann alpha
            R = R_alpha.dot( R_zeta ).dot( R_kappa )

            if dt != adjust.cost.Jet:
                np.testing.assert_allclose( R, adjust.cost.alZeKaToRotationMatrix( angles ) )
        else:
            raise Exception("rotation matrix parametrization not implemented")

        p = objPt - prjCtr

        #fac = -ior[2] / ( p[0]*R[0][2] + p[1]*R[1][2] + p[2]*R[2][2] )
        fac = -ior[2] / p.dot( R[:,2] )
        if adjust.cost.IsInfinite(fac):
            raise Exception("Object point lies in the vanishing plane");

        x0 = ior[0]
        y0 = ior[1]

        # Compute the residual and possibly, also its derivatives
        # Use ``dt(self.obs_x)`` to ensure that the data type (possibly ``Jet``) assigned to ``residuals`` remains unchanged
        #predicted_x = x0 + fac * ( p[0]*R[0][0] + p[1]*R[1][0] + p[2]*R[2][0] )
        predicted_x = x0 + fac * p.dot( R[:,0] )

        residuals[0] = dt(self.obs_x) - predicted_x

        #predicted_y = y0 + fac * ( p[0]*R[0][1] + p[1]*R[1][1] + p[2]*R[2][1] );
        predicted_y = y0 + fac * p.dot( R[:,1] )

        residuals[1] = dt(self.obs_y) - predicted_y

        # indicate success
        return True

class TestCustomCostPhoEvaluate(unittest.TestCase):

    def test_omFiKa(self):
        self._test( adjust.EulerAngles.omfika )

    #def test_fiOmKa(self):
    #    self._test( adjust.EulerAngles.fiomka )

    def test_alZeKa(self):
        self._test( adjust.EulerAngles.alzeka )

    def _test(self, parametrization):
        angles = parameters.EulerAngles( parametrization=parametrization, array=np.array([.1,-.02,.3]) )
        prjCtr = np.array([ 0., 0., 10. ])
        ior    = np.array([ 150., -150., 100. ])
        adp    = parameters.ADP( normalizationRadius=75. )

        point = np.array([  10.,  10., 0. ])
        obs = np.array([ ior[0] +100.+.432, ior[1] +100.+.472 ])

        problem = adjust.Problem()
        loss = adjust.loss.Trivial()
        cost = adjust.cost.PhotoTorlegard( obs[0], obs[1] )
        problem.AddResidualBlock( cost,
                                  loss,
                                  prjCtr,
                                  angles,
                                  ior,
                                  adp,
                                  point )
        residuals, = problem.Evaluate()

        problem = adjust.Problem()
        loss = adjust.loss.Trivial()
        cost = CustomCostPho( obs[0], obs[1], rotMatParametrization=parametrization )
        problem.AddResidualBlock( cost,
                                  loss,
                                  prjCtr,
                                  angles,
                                  ior,
                                  point )
        residuals2, = problem.Evaluate()

        np.testing.assert_allclose( residuals, residuals2 )

class TestCustomCostPho(unittest.TestCase):

    def test_customCostPhoSolve(self):
        angles = np.array([.1,-.02,.3])
        prjCtr = np.array([ 0., 0., 10. ])
        ior    = np.array([ 150., -150., 100. ])

        angles_apri = angles.copy()
        prjCtr_apri = prjCtr.copy()

        points = np.array([ [  10.,  10., 0. ],
                            [ -10.,  10., 0. ],
                            [ -10., -10., 0. ],
                            [  10., -10., 0. ] ] )

        obs = np.array([ [ ior[0] +100.+.432, ior[1] +100.+.472 ],
                         [ ior[0] -100.-.654, ior[1] +100.+.581 ],
                         [ ior[0] -100.+.589, ior[1] -100.+.240 ],
                         [ ior[0] +100.-.487, ior[1] -100.+.541 ] ] )

        loss = adjust.loss.Trivial()

        problem = adjust.Problem()

        for idx in range( points.shape[0] ):
            # 1 residual block per image point
            problem.AddResidualBlock( CustomCostPho( obs[idx][0], obs[idx][1] ),
                                      loss,
                                      prjCtr,
                                      angles,
                                      ior,
                                      points[idx] )
            problem.SetParameterBlockConstant( points[idx] )

        problem.SetParameterBlockConstant( ior )

        residuals_apri, = problem.Evaluate();

        options = adjust.Solver.Options()
        summary = adjust.Solver.Summary()

        adjust.Solve( options, problem, summary )

        self.assertTrue( adjust.isSuccess( summary.termination_type ), "adjustment was unsuccessful" )
        self.assertTrue( summary.final_cost < summary.initial_cost, "adjustment has not reduced the cost" )

        residuals_apost, = problem.Evaluate();

        #print( summary.BriefReport() )
        #print( "prjCtr: {} -> {}".format( prjCtr_apri, prjCtr ) )
        #print( "angles: {} -> {}".format( np.array(angles_apri), np.array(angles) ) )
        #
        #print("residuals:")
        #for idx in range( 0, residuals_apri.shape[0], 2 ):
        #    print( "{} -> {}".format( residuals_apri[idx:idx+2], residuals_apost[idx:idx+2] ) )

        covariance = adjust.Covariance()
        covariance.Compute( [ ( prjCtr, prjCtr ) ], problem )
        covPrjCtr = covariance.GetCovarianceBlock( prjCtr, prjCtr )
        self.assertEqual( covPrjCtr.shape, (3,3) )
        #print( covPrjCtr )

        with self.assertRaises(oriental.Exception):
            covariance.GetCovarianceBlock( angles, prjCtr )

        parNotPartOfProblem = np.zeros(3)
        with self.assertRaises(oriental.Exception):
            covariance.Compute( [ ( parNotPartOfProblem, prjCtr ) ], problem )

class CustomErroneousCost( adjust.cost.AutoDiff ):
    """Cost function that throws an exception"""

    def __init__( self ):
        numResiduals = 1
        parameterSizes = (2,)
        # initialize the base class
        super().__init__( numResiduals, parameterSizes )

    def Evaluate( self, parameters, residuals ):

        # Store the data type of the ``parameters``.
        # It is identical to the data type of ``residuals``
        dt = type(parameters[0][0])
        twoVec = parameters[0]
        thisWillThrowForJet = linalg.norm( twoVec )
        # Compute the residual and possibly, also its derivatives
        # Use ``dt(10.)`` to ensure that the data type (possibly ``Jet``) assigned to ``residuals`` remains unchanged
        residuals[0] = dt(10.) - parameters[0][0]

        # indicate success
        return True

class TestCustomErroneousCost(unittest.TestCase):

    def test_customErroneousCost(self):
        """Let's check if an exception is caught, without terminating the interpreter directly.
        Also, the traceback should include the Python-callstack as OrientAL-Exception-scope"""
        param = np.zeros(2)
        loss = adjust.loss.Trivial()
        problem = adjust.Problem()
        problem.AddResidualBlock( CustomErroneousCost(),
                                  loss,
                                  param )
        problem.Evaluate()
        #try:
        #    problem.Evaluate(jacobian=True)
        #except oriental.Exception as ex:
        #    print(ex)
        self.assertRaises( oriental.Exception, problem.Evaluate, jacobian=True )

class TestParameters(unittest.TestCase):
    
    def test_adp(self):
        self.assertRaises( TypeError, parameters.ADP ) # missing positional argument: normalizationRadius

        adp = parameters.ADP( 1000. )

        adp = parameters.ADP( 1000., adjust.AdpReferencePoint.principalPoint )

        with self.assertRaises( ContractException ):
            adp2 = parameters.ADP( 1000., 1 ) # expected adjust.AdpReferencePoint, not int

        self.assertIsInstance( adp.normalizationRadius, float )
        adp.normalizationRadius = 500.
        adp.normalizationRadius = 100
        with self.assertRaises(ContractException):
            adp.normalizationRadius = 'hallo'
            

        self.assertIsInstance( adp.referencePoint, adjust.AdpReferencePoint )
        adp.referencePoint = adjust.AdpReferencePoint.origin
        with self.assertRaises(ContractException):
            adp.referencePoint = 1 # expected adjust.AdpReferencePoint, not int

    def test_eulerAngles(self):
        angles = parameters.EulerAngles()

        angles = parameters.EulerAngles( parametrization = adjust.EulerAngles.fiomka )

        with self.assertRaises( ContractException ):
            parameters.EulerAngles( 1 ) # expected adjust.EulerAngles, not int

        self.assertIsInstance( angles.parametrization, adjust.EulerAngles )
        angles.parametrization = adjust.EulerAngles.alzeka
        with self.assertRaises(ContractException):
            angles.parametrization = 1 # expected adjust.EulerAngles, not int

class TestLocalParameterization(unittest.TestCase):
    
    def test_getEmpty(self):
        problem = adjust.Problem()
        ior = np.array([0.,0.,0.])
        problem.AddParameterBlock( ior )
        res = problem.GetParameterization( ior )
        self.assertIsNone( res )

    def test_dynamicSubset(self):
        angles = parameters.EulerAngles( parametrization=adjust.EulerAngles.omfika, array=np.array([.1,-.02,.3]) )
        prjCtr = np.array([ 0., 0., 10. ])
        ior    = np.array([ 150., -150., 100. ])
        adp    = parameters.ADP( normalizationRadius=75. )
        obj = np.array([  10.,  10., 0. ])
        img = np.array([ ior[0] +100.+.432, ior[1] +100.+.472 ])
        loss = adjust.loss.Trivial()
        problem = adjust.Problem()

        problem.AddResidualBlock(
            adjust.cost.PhotoTorlegard( img[0], img[1] ),
            loss,
            prjCtr,
            angles,
            ior,
            adp,
            obj )

        jacobian, = problem.Evaluate( residuals=False, jacobian=True )
        self.assertEqual( jacobian.shape[1], 21 )

        subset = adjust.local_parameterization.Subset( 3, [0,1] )
        problem.SetParameterization( ior, subset )

        jacobian, = problem.Evaluate( residuals=False, jacobian=True )
        self.assertEqual( jacobian.shape[1], 21-2 )

        locParam = problem.GetParameterization( ior )
        locParam.setVariable(1)
        problem.ParameterizationLocalSizeChanged( ior )

        jacobian, = problem.Evaluate( residuals=False, jacobian=True )
        self.assertEqual( jacobian.shape[1], 21-1 )

        locParam = problem.GetParameterization( ior )
        locParam.setConstant(1)
        problem.ParameterizationLocalSizeChanged( ior )

        jacobian, = problem.Evaluate( residuals=False, jacobian=True )
        self.assertEqual( jacobian.shape[1], 21-2 )

        # set all variable
        locParam = problem.GetParameterization( ior )
        locParam.setVariable(0)
        locParam.setVariable(1)
        problem.ParameterizationLocalSizeChanged( ior )
        jacobian, = problem.Evaluate( residuals=False, jacobian=True )
        self.assertEqual( jacobian.shape[1], 21 )

        # Using a local parameterization to set all parameters of a block constant is illegal, according to Ceres.
        # However, Ceres checks that only in debug - builds.
        # wk 2016-01-11: I don't remember where Ceres once checked that (it surely does so in the constructor, but not afterwards). Seemingly, it doesn't do so any more.
        #if config.debug:
        #    locParam = problem.GetParameterization( ior )
        #    locParam.setConstant(0)
        #    locParam.setConstant(1)
        #    locParam.setConstant(2)
        #    problem.ParameterizationLocalSizeChanged( ior )
        #    self.assertRaises( oriental.Exception, problem.Evaluate, residuals=False, jacobian=True )
        #    # note that at this point, the problem-object seems to be corrupt:
        #    # re-setting one of the parameters to be variable still results in an error message on the next evaluation!

class TestProblem(unittest.TestCase):
    def test_GetCostFunctionForResidualBlock( self ):
        CostData = namedtuple( 'CostData', 'imgId serial' )
        angles = parameters.EulerAngles( parametrization=adjust.EulerAngles.omfika, array=np.array([.1,-.02,.3]) )
        prjCtr = np.array([ 0., 0., 10. ])
        ior    = np.array([ 150., -150., 100. ])
        adp    = parameters.ADP( normalizationRadius=75. )
        obj = np.array([  10.,  10., 0. ])
        img = np.array([ ior[0] +100.+.432, ior[1] +100.+.472 ])
        loss = adjust.loss.Trivial()
        problem = adjust.Problem()

        cost = adjust.cost.PhotoTorlegard( img[0], img[1] )
        cost.data = CostData('hallo',10)
        resBlock = problem.AddResidualBlock(
            cost,
            loss,
            prjCtr,
            angles,
            ior,
            adp,
            obj )

        cost = problem.GetCostFunctionForResidualBlock( resBlock )
        cost.deAct()
        residualBlocks = problem.GetResidualBlocks()
        cost = problem.GetCostFunctionForResidualBlock( residualBlocks[0] )
        cost.deAct()

    def test_identicalParametersReturned(self):
        angles = parameters.EulerAngles( parametrization=adjust.EulerAngles.omfika, array=np.array([.1,-.02,.3]) )
        prjCtr = np.array([ 0., 0., 10. ])
        ior    = np.array([ 150., -150., 100. ])
        adp    = parameters.ADP( normalizationRadius=75. )
        obj    = parameters.ObjectPoint([  10.,  10., 0. ])
        img = np.array([ ior[0] +100.+.432, ior[1] +100.+.472 ])
        loss = adjust.loss.Trivial()
        problem = adjust.Problem()

        cost = adjust.cost.PhotoTorlegard( img[0], img[1] )
        resBlock = problem.AddResidualBlock(
            cost,
            loss,
            prjCtr,
            angles,
            ior,
            adp,
            obj )

        params = problem.GetParameterBlocksForResidualBlock( resBlock )
        self.assertEqual( 5, len(params) )
        paramIds = sorted( id(el) for el in params )

        paramOrigIds = sorted( id(el) for el in ( prjCtr, angles, ior, adp, obj ) )
        self.assertEqual( paramIds, paramOrigIds )

        objPts = [ el for el in params if type(el)==parameters.ObjectPoint ]
        self.assertEqual( len(objPts), 1 )
        self.assertEqual( id(objPts[0]), id(obj) )


    def test_RemoveResidualBlock(self):
        angless = [ parameters.EulerAngles( parametrization=adjust.EulerAngles.omfika, array=np.array([.1,-.02,.3]) ),
                    parameters.EulerAngles( parametrization=adjust.EulerAngles.omfika, array=np.array([.1,-.03,.3]) ) ]
        prjCtrs = [ np.array([ 0., 0., 10. ]),
                    np.array([ 1., 0., 10. ]) ]
        ior    = np.array([ 150., -150., 100. ])
        adp    = parameters.ADP( normalizationRadius=75. )
        obj = np.array([  10.,  10., 0. ])
        img = np.array([ ior[0] +100.+.432, ior[1] +100.+.472 ])
        loss = adjust.loss.Trivial()
        opts = adjust.Problem.Options()
        opts.enable_fast_removal = True
        problem = adjust.Problem(opts)
        resBlocks = []
        nPts = 100
        nCams = len(prjCtrs)
        for iPt in range(nPts):
            for iCam in range(nCams):
                # random floats in the half-open interval [0.0, 1.0)
                coords = np.random.random(2)
                coords += ior[:2] + 100. - .5
                cost = adjust.cost.PhotoTorlegard( *coords )
                resBlocks.append( problem.AddResidualBlock(
                    cost,
                    loss,
                    prjCtrs[iCam],
                    angless[iCam],
                    ior,
                    adp,
                    obj ) )

        self.assertEqual( problem.NumResidualBlocks() , nPts*nCams  )
        self.assertEqual( problem.NumResiduals()      , nPts*nCams*2  )
        self.assertEqual( problem.NumParameterBlocks(), nCams*2 + 2 + 1 )
        self.assertEqual( problem.NumParameters()     , nCams*6 + 3 + adp.size + 3 )

        residuals = problem.Evaluate()[0].reshape((-1,2))
        self.assertEqual( residuals.shape[0], nPts*nCams )
        residualBlocks = problem.GetResidualBlocks()
        iResId2Remove = 14 # first camera!

        # Any parameters that the residual block depends on are not removed
        problem.RemoveResidualBlock( residualBlocks[iResId2Remove] )

        residuals = np.r_[ residuals[:iResId2Remove],
                           residuals[iResId2Remove+1:] ]
        del residualBlocks[iResId2Remove]
        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.set_residual_blocks( residualBlocks ) # force the same evaluation order!
        residuals2 = problem.Evaluate(evalOpts)[0].reshape((-1,2))
        np.testing.assert_array_almost_equal( residuals, residuals2 )
        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.set_parameter_blocks( [prjCtrs[1],angless[1]] )
        jacobian, = problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        self.assertEqual( jacobian.shape, ( (nPts*nCams-1)*2, 3*2) )
        
        residualBlocks = residualBlocks[1:iResId2Remove:2] + \
                         residualBlocks[iResId2Remove::2]
        evalOpts.set_residual_blocks( residualBlocks ) # ensure the same ordering
        jacobian, = problem.Evaluate( evalOpts, residuals=False, jacobian=True )

        # Any residual blocks that depend on the parameter are also removed
        problem.RemoveParameterBlock( prjCtrs[0] )
        
        self.assertEqual( problem.NumResidualBlocks(), nPts )
        self.assertEqual( problem.NumResiduals(), nPts*2 )
        self.assertEqual( problem.NumParameterBlocks(), 6 ) # angless[0] is still part of the problem
        self.assertEqual( problem.NumParameters(), 5*3 + adp.size )
        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.set_parameter_blocks( [prjCtrs[1],angless[1]] )
        evalOpts.set_residual_blocks( residualBlocks )
        jacobian2, = problem.Evaluate( evalOpts, residuals=False, jacobian=True )
        np.testing.assert_array_almost_equal( jacobian.todense(), jacobian2.todense() )
        dummy=0

class IterationCallback( adjust.IterationCallback ):
    def __init__(self):
        self.data = 5.
        super().__init__()

    def callback( self, iterationSummary ):
        print( self.data )
        return adjust.CallbackReturnType.SOLVER_CONTINUE

class TestCallback(unittest.TestCase):
    def test_userCallbackWithMemberVariable( self ):
        loss = adjust.loss.Trivial()
        block = adjust.Problem()
        param = np.zeros(3)
        covInvSqrt = np.eye(3)
        cost = adjust.cost.NormalPrior( covInvSqrt, param )
        block.AddResidualBlock( cost,
                                loss,
                                param )

        iterationCallback = IterationCallback()
        solveOptions = adjust.Solver.Options()
        solveOptions.callbacks.append( iterationCallback )
        summary = adjust.Solver.Summary()
        adjust.Solve(solveOptions, block, summary)

class TestResidualBlocks(unittest.TestCase):
    def test_identicalHash( self ):
        loss = adjust.loss.Trivial()
        block = adjust.Problem()
        param = np.zeros(3)
        covInvSqrt = np.eye(3)
        cost = adjust.cost.NormalPrior( covInvSqrt, param )
        resBlock = block.AddResidualBlock( cost, loss, param )
        resBlocks = block.GetResidualBlocks()
        self.assertIsNot( resBlock, resBlocks[0], "this is weird, but not important, actually" )
        self.assertEqual( resBlock, resBlocks[0], "2 Python objects that refer to the same residual block in C++ do not compare equal!" )
        self.assertEqual( hash(resBlock), hash(resBlocks[0]), "2 Python objects that refer to the same residual block in C++ do not have the same hash! E.g. set operations with residual block Ids will fail." )
        self.assertIn( resBlock, resBlocks )

if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestAdjustTorlegard.test_redundancyComponents',
                       exit=False )
