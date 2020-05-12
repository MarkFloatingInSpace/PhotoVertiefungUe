# -*- coding: cp1252 -*-
from sys import path
from os.path import dirname, join, realpath
path.append( realpath( join( dirname(__file__), "../../.." ) ) )

import numpy as np

from oriental import adjust
import oriental.adjust.cost
import oriental.adjust.loss
import oriental.adjust.parameters

def example_photo():
    
    angles = adjust.parameters.EulerAngles( parametrization=adjust.EulerAngles.omfika,
                                            array=np.array([.1,-.02,.3]) )
    prjCtr = np.array([ 0., 0., 10. ])
    ior    = np.array([ 150., -150., 100. ])
    adp    = adjust.parameters.ADP( normalizationRadius=75. )
    
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
        problem.AddResidualBlock( adjust.cost.PhotoTorlegard( obs[idx][0], obs[idx][1] ),
                                  loss,
                                  prjCtr,
                                  angles,
                                  ior,
                                  adp,
                                  points[idx] )
        problem.SetParameterBlockConstant( points[idx] )

    problem.SetParameterBlockConstant( ior )
    problem.SetParameterBlockConstant( adp )

    residuals_apri, = problem.Evaluate();

    options = adjust.Solver.Options()
    summary = adjust.Solver.Summary()
    
    adjust.Solve( options, problem, summary )
  
    residuals_apost, = problem.Evaluate();
      
    print( summary.BriefReport() )
    print( "prjCtr: {} -> {}".format( prjCtr_apri, prjCtr ) )
    print( "angles: {} -> {}".format( np.array(angles_apri), np.array(angles) ) )
    
    print("residuals:")
    for idx in range( 0, residuals_apri.shape[0], 2 ):
        print( "{} -> {}".format( residuals_apri[idx:idx+2], residuals_apost[idx:idx+2] ) )
    
    assert summary.final_cost < summary.initial_cost, "adjustment failed"
    
if __name__ == '__main__' :
    example_photo()

   