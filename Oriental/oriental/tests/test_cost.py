# -*- coding: cp1252 -*-
import environment
import oriental
from oriental import adjust
import oriental.adjust.cost
import oriental.adjust.loss
import numpy as np
from copy import copy
import pickle
import unittest
import operator

class TestJet(unittest.TestCase):

    def test_copy(self):
        jet1 = adjust.cost.Jet(2.)
        np.testing.assert_array_equal( jet1.v, np.zeros(adjust.cost.Jet.DIMENSION) )
        jet1_a_orig = jet1.a
        jet1_v_orig = jet1.v.copy()
        jet2 = adjust.cost.Jet(jet1) # copy-construct using the exported copy-constructor
        jet3 = adjust.cost.Jet(3.,1)
        v = np.zeros(adjust.cost.Jet.DIMENSION)
        v[1]=1.
        np.testing.assert_array_equal( jet3.v, v )
        jet2 += jet3
        self.assertEqual( jet1_a_orig, jet1.a )
        np.testing.assert_array_equal( jet1_v_orig, jet1.v )
        self.assertEqual( jet1.a + jet3.a, jet2.a )
        np.testing.assert_array_equal( jet2.v, jet3.v )

        jet2 = copy(jet1) # copy-construct using the copy-module, which uses the __copy__ attribute if present, or pickle-functions otherwise
        jet2 += jet3
        self.assertEqual( jet1_a_orig, jet1.a )
        np.testing.assert_array_equal( jet1_v_orig, jet1.v )
        self.assertEqual( jet1.a + jet3.a, jet2.a )
        np.testing.assert_array_equal( jet2.v, jet3.v )
        dummy=1

    def test_operators(self):
        jet1 = adjust.cost.Jet(2.)
        jet2 = adjust.cost.Jet(3.)
        jetSum = jet1 + jet2
        jetDiff = jet1 - jet2
        jetProd = jet1 * jet2
        jetQuot = jet1 / jet2
        jetSqr = jet1**2
        jetPjet = jet1 ** jet2
        dummy=1

    def test_pickle(self):
        jet1 = adjust.cost.Jet( 3., 1 )
        s = pickle.dumps( jet1 )
        jet2 = pickle.loads( s )
        self.assertEqual( jet1.a, jet2.a )
        np.testing.assert_array_equal( jet1.v, jet2.v )

class TestCost(unittest.TestCase):
    def test_normalPrior(self):
        A = np.arange(6,dtype=float).reshape((2,3))
        b = np.arange(3,dtype=float)
        theCost = adjust.cost.NormalPrior( A, b )
        np.testing.assert_array_equal( theCost.A, A )
        # theCost.A must be read-only!
        self.assertRaises( ValueError, operator.setitem, theCost.A, 0, 1. )

        np.testing.assert_array_equal( theCost.b, b )
        # theCost.b must be read-only!
        self.assertRaises( ValueError, operator.setitem, theCost.b, 0, 1. )

class TestPhotoTorlegard(unittest.TestCase):
    def test_assignUserData(self):
        def assign(data):
            cost1 = adjust.cost.PhotoTorlegard( 100., 200. )
            cost1.data = data
            return cost1
        data = ( 1, 'hello' )
        cost1 = assign(data)
        self.assertEqual( cost1.data, data )
        d = cost1.data
        dummy=1

class TestObservedUnknown(unittest.TestCase):
    def test_evaluate(self):
        weight = np.diag([.1,.1,.1])
        objPt = np.zeros(3)
        cost = adjust.cost.ObservedUnknown( objPt )
        cost = adjust.cost.ObservedUnknown( objPt, weight )
        loss = adjust.loss.Trivial()
        problem = adjust.Problem()
        objPt[0] -= 0.1
        problem.AddResidualBlock( cost,
                                  loss,
                                  objPt )
        residuals, = problem.Evaluate()
        np.testing.assert_array_almost_equal( residuals, np.array([.01, 0.,0.]) )

        # weight must be a square matrix with the same numer of rows as objPt.
        weight = np.array([[1.,1.,1.],
                           [2.,2.,2.]])
        with self.assertRaises(oriental.Exception):
            adjust.cost.ObservedUnknown( objPt, weight )
        weight = np.array([1.,1.,1.])
        with self.assertRaises(oriental.Exception):
            adjust.cost.ObservedUnknown( objPt, weight )

class AnalyticCost( adjust.cost.AnalyticDiff ):
    """Cost function that evaluates `10-x`"""

    def __init__( self, obs ):
        "Initialize an instance of AnalyticCost"

        numResiduals = 1
        parameterSizes = (1,)
        # initialize the base class
        super().__init__( numResiduals, parameterSizes )
        self.obs = obs

    def Evaluate( self, parameters, residuals, jacobians ):
        "Evaluate the ``residuals`` and possibly, their derivatives w.r.t. ``parameters``"

        # Compute the residual
        residuals[0] = self.obs - ( 10. - parameters[0][0] )

        if jacobians and jacobians[0] is not None:
            jacobians[0][0] = 1
        # indicate success
        return True

class TestAnalyticCost(unittest.TestCase):
    def test_analyticCost(self):
        x = np.array([ 5.0 ])
        problem = adjust.Problem()
        loss = adjust.loss.Trivial()
        problem.AddResidualBlock( AnalyticCost(3.), loss, x )
        problem.AddResidualBlock( AnalyticCost(3.5), loss, x )
        residuals, = problem.Evaluate()
        jacobian, = problem.Evaluate(residuals=False, jacobian=True)
        dummy=0

if __name__ == '__main__':
    if not oriental.config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestCost.test_normalPrior',
                       exit=False )
