# -*- coding: cp1252 -*-
"""
Handling of photogrammetric blocks
"""

from oriental import adjust, utils

from contracts import contract, new_contract
import numpy as np

new_contract('Type',type)

def importSubPackagesAndModules():
    "import all sub-packages into the namespace of this package"
    import os.path
    utils.importSubPackagesAndModules( [os.path.dirname(__file__)], 'oriental.blocks' )

@contract
def residualsAndBlocks( block : adjust.Problem,
                        klasses : 'Type|seq[A](Type)' = type(None),
                        forParamBlock : 'None|array[D](float)' = None,
                        weighted : bool = False,
                      ) -> 'tuple(array[BxC](float),list[B])':
    """returns the residuals and residual blocks of `block`, optionally restricted to
      - instances of `klasses`, and/or
      - referencing `forParamBlock`

      if klasses==type(None) is passed, selects residual blocks that have either no data attribute, or a data attribute that is None.
      Pass weighted=True to get residuals divided by their standard deviations, or weighted=False to get the actual residuals.
      Note that `weighted` is respected only by selected cost functions.
    """
    try:
        len(klasses)
    except TypeError:
        klasses = klasses,

    klass2idx = { klass : idx for idx,klass in enumerate(klasses) }
    resBlocksPerClass = [ [] ] * len(klasses)
    for resBlock in ( block.GetResidualBlocks() if forParamBlock is None else block.GetResidualBlocksForParameterBlock( forParamBlock ) ):
        costData = getattr( block.GetCostFunctionForResidualBlock( resBlock ), 'data', None )
        if isinstance( costData, klasses ):
            resBlocksPerClass[ klass2idx[type(costData)] ].append(resBlock)

    allResiduals = []
    allResBlocks = []
    evalOpts = adjust.Problem.EvaluateOptions()
    evalOpts.apply_loss_function = False
    evalOpts.weighted = weighted
    for klass,resBlocks in utils.zip_equal( klasses, resBlocksPerClass ):
        if not len(resBlocks):
            continue
        evalOpts.set_residual_blocks( resBlocks )
        residuals, = block.Evaluate(evalOpts)
        residuals = residuals.reshape(( len(resBlocks), -1 ))
        allResiduals.extend( residuals )
        allResBlocks.extend( resBlocks )

    allResiduals = np.array(allResiduals)
    if not len(allResiduals):
        allResiduals = allResiduals.reshape((-1,2)) # make pycontracts happy
    return allResiduals, allResBlocks

