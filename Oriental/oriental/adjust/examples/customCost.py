# -*- coding: cp1252 -*-
from oriental import adjust
import oriental.adjust.cost
import oriental.adjust.loss

import numpy as np

class CustomCost( adjust.cost.AutoDiff ):
    "Cost function that evaluates `10-x`"
    
    def __init__( self ):
        "Initialize an instance of CustomCost"
        
        numResiduals = 1
        parameterSizes = (1,)
        # initialize the base class
        super().__init__( numResiduals, parameterSizes )
        
    def Evaluate( self, parameters, residuals ):
        "Evaluate the ``residuals`` and possibly, their derivatives w.r.t. ``parameters``"

        # Store the data type of the ``parameters``.
        # It is identical to the data type of ``residuals``
        dt = type(parameters[0][0])
        
        # Compute the residual and possibly, also its derivatives
        # Use ``dt(10.)`` to ensure that the data type (possibly ``Jet``) assigned to ``residuals`` remains unchanged
        residuals[0] = dt(10.) - parameters[0][0]
        
        # indicate success
        return True   

def example_customCost():
    
    # The parameter to find the optimal value for.
    x = np.array([ 5.0 ])
    
    # Copy the original value for comparison with the found optimal value later on.
    x_init = x.copy()

    # Build the problem.
    problem = adjust.Problem()

    # Set up the only cost function (also known as residual).
    cost = CustomCost()
    
    loss = adjust.loss.Trivial()
    
    problem.AddResidualBlock( cost, loss, x )

    options = adjust.Solver.Options()
    
    summary = adjust.Solver.Summary()
    
    # Run the solver!
    adjust.Solve( options, problem, summary )

    print( summary.BriefReport() )
    
    print( "initial value: {}\n"
           "optimal value: {}".format( x_init[0], x[0] ) )
           
    assert summary.final_cost < summary.initial_cost, "adjustment failed"
    
if __name__ == '__main__' :
    example_customCost()
         