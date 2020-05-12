# -*- coding: cp1252 -*-
"block statistics"

import collections
from oriental import adjust, graph, ori, blocks
from contracts import contract
import numpy as np

@contract
def objPtsPerImgAndPerImgPair( block : adjust.Problem, objPts : dict, ImgObsData : type ) -> 'tuple(dict, dict)':
    "returns { imgId : nObj }, { (img1id,img2id) : nObj }"
    imgId2nObj = collections.Counter()
    imgPairIds2nObj = collections.Counter()
    for objPt in objPts.values():
        imgIds = []
        for resBlock in block.GetResidualBlocksForParameterBlock(objPt):
            costData = getattr( block.GetCostFunctionForResidualBlock(resBlock), 'data', None )
            if isinstance( costData, ImgObsData ):
                imgIds.append(costData.imgId)
        for idx,imgId1 in enumerate(imgIds):
            imgId2nObj[imgId1] += 1
            for imgId2 in imgIds[idx+1:]:
                imgPairIds = tuple(sorted([imgId1,imgId2]))
                imgPairIds2nObj[imgPairIds] += 1
    return imgId2nObj, imgPairIds2nObj

@contract
def minCut( imgId2nObj : dict, imgPairIds2nObj : dict ) -> graph.ImageConnectivity.MinCut:
    "returns the minimum cut through the connectivity graph"
    # vertex ids in graph.ImageConnectivity must start at 0 and be consecutive!!
    idx2imgId = { idx : imgId for idx,imgId in enumerate(imgId2nObj) }
    imgId2idx = { imgId : idx for idx,imgId in idx2imgId.items() }
    conn = graph.ImageConnectivity( len(imgId2idx) )
    for (img1id,img2id),nObj in imgPairIds2nObj.items():
        graphEdge = graph.ImageConnectivity.Edge( graph.ImageConnectivity.Image(imgId2idx[img1id]),
                                                  graph.ImageConnectivity.Image(imgId2idx[img2id]),
                                                  float(nObj) )
        conn.addEdgeQuality( graphEdge )
    theMinCut = conn.minCut( returnIndicesSmallerSet=True )
    # hopefully, the type of image IDs fits into!
    #theMinCut.idxsImagesSmallerSet = np.array( [ idx2imgId[int(idx)] for idx in theMinCut.idxsImagesSmallerSet ], theMinCut.idxsImagesSmallerSet.dtype )
    theMinCut.imgIdsSmallerSet = np.array( [ idx2imgId[int(idx)] for idx in theMinCut.idxsImagesSmallerSet ] )
    return theMinCut

