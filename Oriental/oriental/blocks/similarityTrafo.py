# -*- coding: cp1252 -*-
"transform a block based on identical model/object points"

from oriental import adjust, log, ori, utils
import oriental.adjust.cost
import oriental.adjust.loss

from contracts import contract
import numpy as np

logger = log.Logger(__name__)

@contract
def similarityTrafo( ptsModel : 'array[Nx3]', ptsGlobal : 'array[Nx3]', images : dict, objPts : dict, loss = None ):
    loss = loss or adjust.loss.Huber(.1)
    # y=s*R.dot(x-x0)
    # The following minimizes residuals in model space, but that does not matter.
    scale, Rt, P0, sigma0 = ori.similarityTrafo( x=ptsGlobal, y=ptsModel )
    angles = ori.matrix2euler( Rt.T )
    scale = np.array([scale])

    # Let's do a robust adjustment, even though the model points have been triangulated robustly.
    # For that, we must minimize residuals in object space, as that has a meaningful scale.
    #trafoProbl = adjust.Problem()
    #for PRCLocal,PRCGlobal in utils.zip_equal(ptsModel,ptsGlobal):
    #    # P = R.dot( p / s ) + P0
    #    cost = adjust.cost.SimTrafo3d( p=PRCLocal, P=PRCGlobal, parametrization=angles.parametrization )
    #    trafoProbl.AddResidualBlock( cost,
    #                                 loss,
    #                                 P0,
    #                                 angles,
    #                                 scale )
    #options = adjust.Solver.Options()   
    #summary = adjust.Solver.Summary()
    #adjust.Solve(options, trafoProbl, summary)
    #if not adjust.isSuccess(summary.termination_type):
    #    raise Exception("Adjustment of similarity transformation has not converged")
    #evalOpts = adjust.Problem.EvaluateOptions()
    #evalOpts.apply_loss_function = False
    #redundancy = summary.num_residuals_reduced - summary.num_effective_parameters_reduced
    #sigma0 = ( summary.final_cost * 2 / redundancy ) **.5
    #residuals, = trafoProbl.Evaluate(evalOpts)
    #resNormsSqr = residuals[0::3]**2 + residuals[1::3]**2 + residuals[2::3]**2
    #logger.info('Spatial similarity transformation [m]\n'
    #            'statistic\tvalue\n'
    #            'redundancy\t{}\n'
    #            '\N{GREEK SMALL LETTER SIGMA}_0\t{:.4f}\n'
    #            'median residual norm\t{:.4f}\n'
    #            'max. residual norm\t{:.4f}\n',
    #            redundancy, sigma0, np.median(resNormsSqr)**.5, np.max(resNormsSqr)**.5 )

    R = ori.euler2matrix( angles )
    for image in images.values():
        image.prc[:] = P0 + 1./scale * R.dot( image.prc )
        angles = ori.matrix2euler( R.dot( ori.euler2matrix(image.rot) ) )
        image.rot[:] = angles
        image.rot.parametrization = angles.parametrization
    for objPt in objPts.values():
        objPt[:] = P0 + 1./scale * R.dot( objPt )