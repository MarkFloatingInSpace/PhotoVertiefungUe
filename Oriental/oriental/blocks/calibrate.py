# -*- coding: cp1252 -*-
"calibrate cameras"
from oriental import adjust, log, blocks
import oriental.blocks.qxx
import oriental.adjust.local_parameterization
import oriental.adjust.parameters

from contracts import contract
import numpy as np

import collections
from itertools import chain

logger = log.Logger(__name__)

@contract
def makeIorsAdpsVariableAtOnce( block : adjust.Problem,
                                solveOpts : adjust.Solver.Options,
                                cameras : dict,
                                otherParsToCheck : 'seq' = [],
                                params : 'seq[>0]' = ( adjust.PhotoDistortion.optPolynomRadial3, 2, 0, adjust.PhotoDistortion.optPolynomRadial5 ) ):
    # extend the set of estimated ior/adp parameters
    atOnce = True
    iorIds = list(cameras)
    iIorId = 0
    iLastSuccessParam=0
    iLastSuccessIorId=0
    iParam = 0
    summary = None
    while 1:
        if atOnce:
            msgs = makeIorAdpVariableAtOnce( block, cameras, otherParsToCheck, params[iParam] )
            if msgs:
                iLastSuccessParam = iParam
            else:
                iParam += 1
                if iParam == len(params):
                    iParam=0
                if iLastSuccessParam == iParam:
                    if len(iorIds)<2:
                        break
                    logger.verbose('Switching makeIorAdpVariableAtOnce to single-ior/adp mode')
                    atOnce = False
                    iParam = 0
                    iLastSuccessParam = 0
                continue
        else:
            msgs = makeIorAdpVariableAtOnce( block, cameras, otherParsToCheck, params[iParam], iorId=iorIds[iIorId] )
            if msgs:
                iLastSuccessParam=iParam
                iLastSuccessIorId=iIorId
            iIorId += 1
            if iIorId == len(iorIds):
                iIorId=0
                iParam += 1
                if iParam == len(params):
                    iParam=0
            if not msgs:
                if iLastSuccessParam==iParam and iLastSuccessIorId==iIorId:
                    break
                continue
        logger.info( '\n'.join(msgs) )
        logger.verbose("Full adjustment ...")
        summary = adjust.Solver.Summary()
        adjust.Solve(solveOpts, block, summary)
        logger.info("Full adjustment done.")
        
        if not adjust.isSuccess( summary.termination_type ):
            # this state could be handled by removing the added observations from the block again.
            logger.info( summary.FullReport() )
            raise Exception("adjustment failed after additional parameters have been introduced into the block")
    return summary

@contract
def makeIorAdpVariableAtOnce( block : adjust.Problem,
                              cameras : dict,
                              otherParsToCheck,
                              param, # ior param index or adjust.PhotoDistortion; for ior, index 0 or 1 are both treated as 'principal point'
                              iorId : 'int|None' = None,
                              maxAbsCorr : float = 0.7
                            ) -> 'list(str)':
    # Try to set variable at once as many ior/adp parameters as possible and reasonable.
    # Need to pass all and only non-constant parameter blocks to adjust.sparseQxx

    # Introduce r3, z0, (x0,y0), r5 one after another, but for all cameras at the same time.
    # This may not yield the maximum set of variable ior/adp parameters. Thus, call makeIorAdpVariable after this function has 'converged'.
    msgs = []

    def isIor(par):
        return type(par) == int

    paramName = adjust.parameters.ADP(1.).names[param] if not isIor(param) else 'focal length' if param==2 else 'principal point'
    params = [0,1] if isIor(param) and param in (0,1) else [param]

    Intrinsic = collections.namedtuple( 'Intrinsic', [ 'parBlock', 'wasConst', 'subset' ] )
    IntrinsicPair = collections.namedtuple( 'IntrinsicPair', [ 'parBlock', 'wasConst', 'subset', 'anyParsSetFree', 'oParBlock', 'oIsConst', 'oSubset' ] )

    def getIntrinsicPairs():
        intrinsicPairs = collections.OrderedDict()
        for camera in cameras.values():
            #if img.isCalibrated:
            #    continue # avoid zero-columns in jacobian!
            intrinsicPair = intrinsicPairs.get(camera.id)
            if intrinsicPair:
                continue
            intrinsic = Intrinsic( camera.ior, block.IsParameterBlockConstant( camera.ior ), block.GetParameterization( camera.ior ) )
            oIntrinsic = Intrinsic( camera.adp, block.IsParameterBlockConstant( camera.adp ), block.GetParameterization( camera.adp ) )
            if not isIor(param):
                intrinsic, oIntrinsic = oIntrinsic, intrinsic

            anyParsSetFree = False
            if iorId in ( None, camera.id ):
                if intrinsic.wasConst:
                    block.SetParameterBlockVariable(intrinsic.parBlock)
                    anyParsSetFree = True

                if not intrinsic.subset:
                    if intrinsic.wasConst:
                        # set all parameters constant except for the first in coeffs
                        locPar = adjust.local_parameterization.Subset( intrinsic.parBlock.size, [ el for el in range(intrinsic.parBlock.size) if el not in params ] )
                        intrinsic = intrinsic._replace(subset=locPar)
                        block.SetParameterization( intrinsic.parBlock, locPar )
                        anyParsSetFree = True
                else:
                    if intrinsic.wasConst:
                        # We must wipe all other free parameters from subset: e.g. adjustment of ior's focal length has been tried before, but failed, and was set constant, again. Now, we may want to estimate the principal point
                        wantedConstancyMask = np.ones_like( intrinsic.subset.constancyMask )
                        wantedConstancyMask[params] = 0
                    else:
                        wantedConstancyMask = intrinsic.subset.constancyMask.copy()
                        wantedConstancyMask[params] = 0
                    iDiffs = np.flatnonzero( intrinsic.subset.constancyMask != wantedConstancyMask )
                    if iDiffs.size:
                        for iDiff in iDiffs:
                            if wantedConstancyMask[iDiff]:
                                intrinsic.subset.setConstant( int(iDiff) )
                            else:
                                intrinsic.subset.setVariable( int(iDiff) )
                        block.ParameterizationLocalSizeChanged( intrinsic.parBlock )
                        anyParsSetFree = True

            intrinsicPairs[camera.id] = IntrinsicPair( *chain( intrinsic, [anyParsSetFree], oIntrinsic ) )

        return intrinsicPairs

    def logOrRevert( intrinsicPairs, maxAbsCorrsSqr ):
        # for all parameters/parameter blocks with correlations above maxAbsCorr, revert the changes from above. For the others, produce log messages.
        # Once a parameter has been set variable, it shall never be set constant, again.
        maxAbsCorrSqr = maxAbsCorr**2
        iPar = 0
        for currIorID,intrinsicPair in intrinsicPairs.items():
            # do not re-set parameters to constant that were set free in preceding function calls
            if iorId in ( None, currIorID ):
                nVariable = intrinsicPair.subset.constancyMask.size - intrinsicPair.subset.constancyMask.sum()
                if intrinsicPair.anyParsSetFree:
                    # check only the parameter(s) under question, but not all of non-const subset
                    offsets = np.cumsum( np.logical_not(intrinsicPair.subset.constancyMask) ) - 1
                    idxs = offsets[ params ]
                    if maxAbsCorrsSqr is not None and maxAbsCorrsSqr[iPar + idxs].max() <= maxAbsCorrSqr:
                        msgs.append( 'Free parameter {} for iorId {}'.format( paramName, currIorID ) )
                    elif nVariable - len(params) == 0:
                        # Setting all parameters of a block to constant is illegal. Instead, set the whole block constant!
                        block.SetParameterBlockConstant( intrinsicPair.parBlock )
                    else:
                        for coeff in params:
                            intrinsicPair.subset.setConstant( coeff )
                        block.ParameterizationLocalSizeChanged( intrinsicPair.parBlock )
            else:
                if intrinsicPair.wasConst: # in this case, the par block has not been freed!
                    nVariable = 0
                elif intrinsicPair.subset:
                    nVariable = intrinsicPair.subset.constancyMask.size - intrinsicPair.subset.constancyMask.sum()
                else:
                    nVariable = intrinsicPair.parBlock.size

            iPar += nVariable

            if not intrinsicPair.oIsConst:
                if intrinsicPair.oSubset:
                    iPar += intrinsicPair.oSubset.constancyMask.size - intrinsicPair.oSubset.constancyMask.sum()
                else:
                    iPar += intrinsicPair.oParBlock.size

    intrinsicPairs = getIntrinsicPairs()
    if not any( el.anyParsSetFree for el in intrinsicPairs.values() ):
        return msgs # all cameras had the current parameters already set free, nothing to do.

    maxAbsCorrsSqr = None
    try:
        RxxSqr, *_ = blocks.qxx.RxxSqr( block,
                                        cameras.values(),
                                        otherParsToCheck )
    except adjust.ExcCholmod:
        pass
    else:
        maxAbsCorrsSqr = RxxSqr.max( axis=1 )
    finally:
        logOrRevert( intrinsicPairs, maxAbsCorrsSqr )
    return msgs
