# -*- coding: cp1252 -*-
import environment
import oriental
from oriental import config, adjust, log
import oriental.adjust.loss
import oriental.adjust.cost
from oriental.adjust import parameters

adjust.setCeresMinLogLevel( log.Severity.info )

import numpy as np
import unittest

class TestErroneousUserDefinedLossFunction(unittest.TestCase):

    def setUp(self):
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


        self.problem = adjust.Problem()

        self.options = adjust.Solver.Options()
        self.summary = adjust.Solver.Summary()

    def test_trivialAsCustom(self):

        class Trivial( adjust.loss.IUser ):
            def __init__( self ):
                super().__init__()

            def Evaluate( self, sq_norm, d ):
                d[0]=sq_norm
                d[1]=1.
                d[2]=0.

        loss = Trivial()

        for idx in range( self.points.shape[0] ):
            # 1 residual block per image point
            self.problem.AddResidualBlock( adjust.cost.PhotoTorlegard( *self.obs[idx] ),
                                      loss,
                                      self.prjCtr,
                                      self.angles,
                                      self.ior,
                                      self.adp,
                                      self.points[idx] )
            self.problem.SetParameterBlockConstant( self.points[idx] )

        self.problem.SetParameterBlockConstant( self.ior )
        self.problem.SetParameterBlockConstant( self.adp )

        # must not throw
        adjust.Solve( self.options, self.problem, self.summary )

    def test_baseNotInitialized(self):
        # custom loss functor whose constructor does not initialize its base class -> usage in `Solve` shall raise an exception, but not crash the interpreter

        class MyLoss( adjust.loss.IUser ):
            def __init__( self ):
                # super().__init__()
                pass

        loss = MyLoss()

        try:
            self.problem.AddResidualBlock( adjust.cost.PhotoTorlegard( *self.obs[0] ),
                                           loss,
                                           self.prjCtr,
                                           self.angles,
                                           self.ior,
                                           self.adp,
                                           self.points[0] )

        except TypeError as ex:
            print(ex)
        self.assertRaises( TypeError, self.problem.AddResidualBlock,
                                      adjust.cost.PhotoTorlegard( *self.obs[0] ),
                                      loss,
                                      self.prjCtr,
                                      self.angles,
                                      self.ior,
                                      self.adp,
                                      self.points[0] )
    def test_overloadMissing(self):
        "custom loss functor missing the overload of Evaluate -> instantiation shall raise an exception, but not crash the interpreter"

        class MyLoss( adjust.loss.IUser ):
            def __init__( self ):
                super().__init__()


        try:
            loss = MyLoss()
        except oriental.Exception as ex:
            print(ex)
        self.assertRaises( oriental.Exception, MyLoss )

    def test_wrongArity(self):
        # overload with wrong number of arguments -> instantiation shall raise an exception, but not crash the interpreter

        class MyLoss( adjust.loss.IUser ):
            def __init__( self ):
                super().__init__()

            def Evaluate( self, sq_norm ):
                dummy = 1

        self.assertRaises( oriental.Exception,
                           MyLoss )

    def test_firstOrderDerivativeIsZero(self):
        # assign zero to rho' -> usage in `Solve` shall raise an exception, but not crash the interpreter

        class MyLoss( adjust.loss.IUser ):
            def __init__( self ):
                super().__init__()

            def Evaluate( self, sq_norm, d ):
                d[0]=0
                d[1]=0 # rho' <= 0 is illegal. Starting from ceres-1.9, rho' <= 0 is only checked for residuals != 0 with rho'' < 0.
                d[2]=1 #                       Thus, set rho'' = 1
                dummy = 1

        loss = MyLoss()

        for idx in range( self.points.shape[0] ):
            # 1 residual block per image point
            self.problem.AddResidualBlock( adjust.cost.PhotoTorlegard( *self.obs[idx] ),
                                           loss,
                                           self.prjCtr,
                                           self.angles,
                                           self.ior,
                                           self.adp,
                                           self.points[idx] )
            self.problem.SetParameterBlockConstant( self.points[idx] )

        self.problem.SetParameterBlockConstant( self.ior )
        self.problem.SetParameterBlockConstant( self.adp )

        #try:
        #    adjust.Solve( self.options, self.problem, self.summary )
        #except oriental.Exception as ex:
        #    print(ex)
        self.assertRaises( oriental.Exception,
                           adjust.Solve, self.options, self.problem, self.summary )


    def test_assignNonConvertibleType(self):
        # assign unsupported type to rho' -> usage in `Solve` shall raise an exception, but not crash the interpreter

        class MyLoss( adjust.loss.IUser ):
            def __init__( self ):
                super().__init__()

            def Evaluate( self, sq_norm, d ):
                d[0]=0
                d[1]="hello"
                d[2]=0
                dummy = 1

        loss = MyLoss()

        for idx in range( self.points.shape[0] ):
            # 1 residual block per image point
            self.problem.AddResidualBlock( adjust.cost.PhotoTorlegard( *self.obs[idx] ),
                                      loss,
                                      self.prjCtr,
                                      self.angles,
                                      self.ior,
                                      self.adp,
                                      self.points[idx] )
            self.problem.SetParameterBlockConstant( self.points[idx] )

        self.problem.SetParameterBlockConstant( self.ior )
        self.problem.SetParameterBlockConstant( self.adp )

        #try:
        #    adjust.Solve( self.options, self.problem, self.summary )
        #except oriental.Exception as ex:
        #    print(ex)
        self.assertRaises( oriental.Exception, adjust.Solve, self.options, self.problem, self.summary )


if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestErroneousUserDefinedLossFunction.test_firstOrderDerivativeIsZero',
                       exit=False )
