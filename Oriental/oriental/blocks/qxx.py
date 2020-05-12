# -*- coding: cp1252 -*-
"get the stochastic model a posteriori"
from oriental import adjust, log
import oriental.adjust.local_parameterization

import collections.abc

from contracts import contract
import numpy as np

logger = log.Logger(__name__)

@contract
def RxxSqr( block : adjust.Problem,
            parBlocks1, # iterable of parameter blocks whose correlations we are interested in
            parBlocks2 = [] # other iterable of parameter blocks whose correlations we are interested in.
          ):
    """returns a tuple of:
       - the element-wise squared correlation matrix (diagonal is zero) for the non-constant parameter blocks of parBlocks1 + parBlocks2.
       - the non-constant parameter blocks of parBlocks1 + parBlocks2
       - the number of non-constant parameter blocks of parBlocks1

       This function takes care to use all non-constant blocks of `block` in the computation of the Jacobian.
       As it returns the square of Rxx, the caller may completely avoid taking square roots when comparing a certain upper threshold to the maximum correlation of a parameter with another parameter:
       compare the square of the threshold to the maximum of the squared correlation coefficients.
    """
    varParBlocks = [ el for el in parBlocks1 if not block.IsParameterBlockConstant(el) ]
    nVarParBlocks1 = len(varParBlocks)
    varParBlocks.extend(( el for el in parBlocks2 if not block.IsParameterBlockConstant(el) ))
    nVarWantedParBlocks = len(varParBlocks)
    varWantedParBlockIds = frozenset( id(el) for el in varParBlocks )
    if len(varWantedParBlockIds) != nVarWantedParBlocks:
        raise Exception('parBlocks1 + parBlocks2 contains duplicates.')
    nPars2Skip = 0
    for el in block.GetParameterBlocks():
        if id(el) not in varWantedParBlockIds and not block.IsParameterBlockConstant(el):
            varParBlocks.append(el)
            nPars2Skip += block.ParameterBlockLocalSize(el)
    del varWantedParBlockIds

    evalOpts = adjust.Problem.EvaluateOptions()
    evalOpts.apply_loss_function = False
    evalOpts.weighted = True
    evalOpts.set_parameter_blocks( varParBlocks )
    # jacobian contains columns only for varParBlocks
    # jacobian contains no columns for parameters that are set constant by way of a Subset-parameterization
    jacobian, = block.Evaluate( evalOpts, residuals=False, jacobian=True )
    assert jacobian.shape[1] == np.unique( jacobian.nonzero()[1] ).size, "zero columns in jacobian"
    nPars2Evaluate = jacobian.shape[1] - nPars2Skip
    # A.T.dot(A) may not be invertible!
    if 1:
        # returns an upper triangular crs matrix
        # TODO: simplicial factorization (which is selected automatically for small problems) crashes with a segmentation fault
        QxxAll = adjust.sparseQxx( jacobian, adjust.Factorization.supernodal )
        # column slicing is very inefficient on csr matrices. Thus, convert to csc
        Qxx = QxxAll[:nPars2Evaluate,:].tocsc()[:,:nPars2Evaluate]
        # TODO: Not even the rows and columns concerning ior/adp seem to be dense, but only the whole diagonal.
        # Is that only the case for supernodal factorization?
        #if Qxx.nnz != Qxx.shape[0]*(Qxx.shape[0]+1)/2:
        #    import pickle
        #    with open('qxx.pickle','wb') as fout:
        #        pickle.dump( Qxx, fout, protocol=pickle.HIGHEST_PROTOCOL )
        #    raise Exception( 'sub-matrix for cameras is not dense! Qxx dumped to file: {}'.format('qxx.pickle') )
        Qxx = Qxx.toarray()
        diagQxx = Qxx.diagonal().copy()
        Rxx = Qxx + Qxx.T
    else:
        from scipy import linalg
        #jacobianDense = jacobian.toarray()
        # N = jacobianDense.T.dot( jacobianDense )
        N = jacobian.transpose().dot(jacobian).toarray()
        del jacobian
        #del block
        import gc
        gc.collect()
        logger.info('N computed')
        C,lower = linalg.cho_factor(N,overwrite_a=True)
        del N
        logger.info('C computed. Shape: {} x {} rows x cols', *C.shape)
        unitVecs = np.eye( C.shape[0], C.shape[1]-nPars2Skip )
        logger.info('unitVecs allocated')
        QxxAll = linalg.cho_solve( (C,lower), unitVecs, overwrite_b=True )
        logger.info('unitVecs inverted')
        Qxx = QxxAll[:-nPars2Skip,:]
        #QxxAll = linalg.inv( N )
        #logger.info('Qxx computed')
        #Qxx = QxxAll[:-nPars2Skip, :-nPars2Skip]
        diagQxx = Qxx.diagonal().copy()
        Rxx = Qxx
    # If nPars2print is much smaller than nPars2Evaluate, then it pays off to not compute square roots for all of diagQxx.
    # Instead, square the off-diagonal elements and compute the squares of the correlation coefficients. That way, we don't need to take the absolute value explicitly.
    # Equivalent to:
    # diagQxxSqrt = diagQxx ** .5
    # Rxx = ( ( Rxx / diagQxxSqrt ).T / diagQxxSqrt ).T
    # maxAbsCorrs = np.abs(Rxx).max(axis=1)
    np.fill_diagonal(Rxx,0)
    RxxSqr = ( ( Rxx ** 2 / diagQxx ).T / diagQxx ).T
    return RxxSqr, varParBlocks[:nVarWantedParBlocks], nVarParBlocks1

@contract
def diagQxxPerParBlock( block : adjust.Problem, wantedParBlocks : collections.abc.Iterable = [] ) -> dict:
    """return the diagonal of Qxx for each of `wantedParBlocks`
       constant `wantedParBlocks` get np.zeros of appropriate size.
       parameter blocks not in `wantedParBlocks` are omitted in the returned dict.
    """
    def splitVarAndConst(parBlocks):
        varBlocks = []
        constBlocks = []
        for parBlock in parBlocks:
            if block.IsParameterBlockConstant(parBlock):
                constBlocks.append(parBlock)
            else:
                varBlocks.append(parBlock)
        return varBlocks, constBlocks
    varParBlocks, constWantedParBlocks = splitVarAndConst(wantedParBlocks)
    returnAll = len(varParBlocks) + len(constWantedParBlocks) == 0
    if returnAll:
        varParBlocks, constWantedParBlocks = splitVarAndConst( block.GetParameterBlocks() )
    nVarWantedParBlocks = len(varParBlocks)

    def appendOtherVarParBlocks(varParBlocks, block):
        varWantedParBlockIds = frozenset( id(el) for el in varParBlocks )
        if len(varWantedParBlockIds) != len(varParBlocks):
           raise Exception('wantedParBlocks contains duplicates')
        for parBlock in block.GetParameterBlocks():
            if id(parBlock) not in varWantedParBlockIds and not block.IsParameterBlockConstant(parBlock):
                varParBlocks.append(parBlock)
    if not returnAll: # in this case, the following is superfluous
        appendOtherVarParBlocks( varParBlocks, block )

    evalOpts = adjust.Problem.EvaluateOptions()
    evalOpts.apply_loss_function = False
    evalOpts.weighted = True
    evalOpts.set_parameter_blocks( varParBlocks )
    # jacobian contains columns only for varParBlocks
    # jacobian contains no columns for parameters that are set constant by way of a Subset-parameterization
    jacobian, = block.Evaluate( evalOpts, residuals=False, jacobian=True )
    diagQxx = adjust.diagQxx( jacobian, adjust.Factorization.supernodal )
    cofactors = {}
    iPar = 0
    for parBlock in varParBlocks[:nVarWantedParBlocks]:
        locPar = block.GetParameterization( parBlock )
        if locPar is None:
            dQxx = diagQxx[ iPar : iPar+parBlock.size ]
            iPar += parBlock.size
        elif isinstance( locPar, adjust.local_parameterization.Subset ):
            bVariable = np.logical_not(locPar.constancyMask)
            dQxx = np.zeros( parBlock.size )
            dQxx[bVariable] = diagQxx[ iPar : iPar+bVariable.sum() ]
            iPar += bVariable.sum()
        else:
            # TODO do as in CovarianceImpl::GetCovarianceBlockInTangentOrAmbientSpace.
            # We'd need the off-diagonal elements of Qxx for that -> implement new function adjust.diagQxxBlocks, which returns { parBlock.ctypes.data : diagQxxBlock }
            locSize = locPar.LocalSize()
            dQxx = np.zeros( parBlock.size )
            dQxx[:locSize] = diagQxx[ iPar : iPar+locSize ]
            iPar += locSize

        # dtype=np.float must be passed, otherwise: ndarray.dtype==object
        cofactors[parBlock.ctypes.data] = dQxx

    # For all-constant parameter blocks, store zero-std.devs instead of NULLs, because 0.0 means constant, while NULL means 'sigma not estimated'
    for parBlock in constWantedParBlocks:
        cofactors[parBlock.ctypes.data] = np.zeros(parBlock.size)

    return cofactors
