# -*- coding: cp1252 -*-
"deactivate observations"
from pathlib import Path
from oriental import adjust, log, ori, blocks
from oriental.utils import zip_equal

from contracts import contract
import numpy as np
from scipy import linalg

class DeactivateTiePts:
    """Remove tie image point observations, tie object points, images, and cameras from the block.
    Removes no other observation types.
    Removes tie object points, images, and cameras only if observed exclusively by tie image points.
    """
    @contract
    def __init__( self, block : adjust.Problem,
                        solveOpts : adjust.Solver.Options,
                        cameras : dict,
                        images : dict,
                        tieObjPts : dict,
                        tieImgObsDataType : type = type(None),
                        minResBlocksPerPho : int = 20,
                        minResBlocksPerPoint : int = 2):
        self.block = block
        self.solveOpts = solveOpts
        self.cameras = cameras
        self.images = images
        self.tieObjPts = tieObjPts
        self.TieImgObsData = tieImgObsDataType
        self.minResBlocksPerPho = minResBlocksPerPho
        self.minResBlocksPerPoint = minResBlocksPerPoint
        self.logger = log.Logger(__name__ + ".DeactivateTiePts")

    @contract
    def deacBehindCam( self, maxOffAxisAngleGon : 'float,>0,<100' = 99. ) -> 'array[2](int,>-1)':
        resBlocks2remove = []
        # tan(alpha)    == b    / a
        # tan(alpha)**2 == |xy|**2 / z**2
        maxAngleTanSqr = np.tan(maxOffAxisAngleGon / 200. * np.pi)**2 # avoid taking square roots!
        for img in self.images.values():
            R = ori.euler2matrix( img.rot )
            # If Problem::Options::enable_fast_removal is true, then getting the residual blocks is fast and depends only on the number of residual blocks.
            # Otherwise, getting the residual blocks for a parameter block will incur a scan of the entire Problem object.
            for resBlock in self.block.GetResidualBlocksForParameterBlock( img.prc ):
                costData = getattr( self.block.GetCostFunctionForResidualBlock(resBlock), 'data', None )
                if not isinstance( costData, self.TieImgObsData ):
                    continue
                objPt = self.tieObjPts[costData.objPtId]
                # Do not simply test ptCam[2] against 0.
                #ptCamZ = R[:,2].dot( objPt - img.prc )
                ptCam = R.T.dot( objPt - img.prc )
                if ptCam[2] > 0:
                    resBlocks2remove.append(resBlock)
                    continue # behind camera
                ptXYSqr = np.sum(ptCam[:2]**2)
                ptZSqr = ptCam[2]**2
                if ptXYSqr > maxAngleTanSqr * ptZSqr:
                    resBlocks2remove.append(resBlock)
        nDeacs = self._deac(resBlocks2remove)
        if nDeacs.any():
            self.logger.info(f'Removed {nDeacs[0]} imgPts, {nDeacs[1]} objPts: off-axis angle > {maxOffAxisAngleGon} gon')
        return nDeacs

    @contract
    def deacLargeResidualNorms( self, maxTieImgResNorm : '>0',
                                      residuals : 'array(float)|None' = None,
                                      weighted : bool = False ) -> 'array[2](int,>-1)':
        """computes residuals and deactivates costs whose residuals are larger than maxTieImgResNorm.
           pass weighted=False to compute and evaluate unweighted residuals. maxTieImgResNorm should be adapted to that.
           stores the computed residuals in-place.
        """
        maxTieImgResNormSqr = maxTieImgResNorm**2

        residuals_, resBlocks = blocks.residualsAndBlocks( self.block, self.TieImgObsData, weighted=weighted )
        resNormsSqr = np.sum(residuals_ ** 2, axis=1)
        keep = resNormsSqr <= maxTieImgResNormSqr
        resBlocks2remove = [resBlock for idx, resBlock in enumerate(resBlocks) if not keep[idx]]

        nDeacs = self._deac(resBlocks2remove)
        if nDeacs.any():
            self.logger.info(f'Removed {nDeacs[0]} imgPts, {nDeacs[1]} objPts: {"weighted" if weighted else ""} image residual norms > {maxTieImgResNorm}')

        if residuals is not None:
            # This is a bit dirty: inplace-resize the array, telling numpy to not check the reference count (if the array-data is referenced by a sub-array, then that may lead to a crash).
            # Since, if passed in, residuals's reference count is at least 2, the reference count check would always fail.
            # The alternative would be to return residuals. However, this would mean that deacLargeResidualNorms returns more than just nDeacs, as deacBehindCam and deacSmallAngle do.
            residuals.resize( (np.sum(keep), *residuals_.shape[1:]), refcheck=False )
            residuals[:] = residuals_[keep]

        return nDeacs

    @contract
    def deacSmallAngle( self, angleThreshGon : '>0,<100' ) -> 'array[2](int,>-1)':
        cosMinAngle = np.cos( angleThreshGon / 200. * np.pi )
        cosMaxAngle = -cosMinAngle
        objPtIds2remove = []
        def checkAngle( objPtId, objPt ):
            resBlocks = [ resBlock for resBlock in self.block.GetResidualBlocksForParameterBlock( objPt )
                          if isinstance( getattr( self.block.GetCostFunctionForResidualBlock(resBlock), 'data', None ), self.TieImgObsData ) ]
            # We expect the vast majority of objPts to have large enough intersection angles.
            # Upon encountering the first angle that seems okay, we exit early.
            # Thus, do not pre-compute all unit vectors, but compute them on-demand,
            # taking advantage of unit vectors being needed from idx zero upwards consecutively.
            unitVecs = []
            def getUnitVec(idx):
                if idx == len(unitVecs):
                    cost = self.block.GetCostFunctionForResidualBlock(resBlocks[idx])
                    img = self.images[cost.data.imgId]
                    d = objPt - img.prc
                    unitVecs.append( d / linalg.norm(d) )
                return unitVecs[idx]

            for iResBlock1 in range(len(resBlocks)-1):
                d1 = getUnitVec(iResBlock1)
                for iResBlock2 in range(iResBlock1+1, len(resBlocks)):
                    d2 = getUnitVec(iResBlock2)
                    cosIntersectAngle = d1.dot(d2)
                    if cosIntersectAngle <= cosMinAngle and cosIntersectAngle >= cosMaxAngle:
                        return
            objPtIds2remove.append(objPtId)
                
        for objPtId,objPt in self.tieObjPts.items():
            checkAngle( objPtId, objPt )
        nDeacs = self._deac( None, objPtIds2remove)
        if nDeacs.any():
            self.logger.info(f'Removed {nDeacs[0]} imgPts, {nDeacs[1]} objPts: all intersections outside [{angleThreshGon};{200-angleThreshGon}] gon')
        return nDeacs

    @contract
    def deacCloseToPrc(self, minDist : 'float,>0') -> 'array[2](int,>-1)':
        'Of course, this deactivation method depends on the scale of the block -> transform to metric CS before.'
        minDistSqr = minDist**2
        objPtIds2remove = []
        for resBlock in self.block.GetResidualBlocks():
            costData = getattr(self.block.GetCostFunctionForResidualBlock(resBlock), 'data', None)
            if not isinstance(costData, self.TieImgObsData):
                continue
            image = self.images[costData.imgId]
            objPt = self.tieObjPts[costData.objPtId]
            if np.sum((objPt - image.prc)**2) < minDistSqr:
                objPtIds2remove.append(costData.objPtId)
        nDeacs = self._deac(None, objPtIds2remove)
        if nDeacs.any():
            self.logger.info( f'Removed {nDeacs[0]} imgPts, {nDeacs[1]} objPts: closer to PRC than {minDist}' )
        return nDeacs

    @contract
    def deacFewImgResBlocks(self) -> 'array[2](int,>-1)':
        nDeacs = self._deac(affectedImgIds = list(self.images))
        if nDeacs.any():
            self.logger.info(f'Removed {nDeacs[0]} imgPts, {nDeacs[1]} objPts: images with less than {self.minResBlocksPerPho} tie image points and no other residual blocks.')
        return nDeacs

    @contract
    def _deac( self, resBlocks2remove : 'None|seq' = None, objPtIds2remove : 'None|list' = None, affectedImgIds : 'None|list' = None ) -> 'array[2](int,>-1)':
        # n.b.: the passed arguments are mutable. Thus, changing the in-place may affect outer function calls. However, the implementation takes care of this.
        resBlocks2remove = resBlocks2remove or []
        objPtIds2remove = objPtIds2remove or []
        affectedImgIds = affectedImgIds or []
        nDeacs = np.zeros(2,np.int)

        if not any((resBlocks2remove, objPtIds2remove, affectedImgIds)):
            return nDeacs

        for resBlock in resBlocks2remove:
            costData = getattr( self.block.GetCostFunctionForResidualBlock(resBlock), 'data', None )
            assert isinstance( costData, self.TieImgObsData )
            affectedImgIds.append( costData.imgId )
            self.block.RemoveResidualBlock( resBlock )
            nDeacs[0] += 1
            resBlocks = self.block.GetResidualBlocksForParameterBlock( self.tieObjPts[costData.objPtId] )
            if len(resBlocks) < self.minResBlocksPerPoint and \
                all( isinstance( getattr( self.block.GetCostFunctionForResidualBlock(resBlock), 'data', None ),
                                 self.TieImgObsData ) for resBlock in resBlocks ):
                objPtIds2remove.append( costData.objPtId )
        del resBlocks2remove

        for objPtId in frozenset(objPtIds2remove):
            objPt = self.tieObjPts[objPtId]
            removeObjPt = True
            for resBlock in self.block.GetResidualBlocksForParameterBlock( objPt ):
                costData = getattr( self.block.GetCostFunctionForResidualBlock(resBlock), 'data', None )
                if isinstance( costData, self.TieImgObsData ):
                    # Problem.RemoveParameterBlock(objPt) removes not only the parameter block, but also all residual blocks that depend on it.
                    # As we do not check beforehand if objPt shall be removed or not (because it also depends on other observation types), we remove the tie image point observations one-by-one ourself.
                    self.block.RemoveResidualBlock( resBlock )
                    nDeacs[0] += 1
                    affectedImgIds.append( costData.imgId )
                else:
                    # we keep objPts that (additionally) depend on other observation types. But still, we remove the tie image point obs.
                    removeObjPt = False
            if removeObjPt:
                del self.tieObjPts[objPtId]
                nDeacs[1] += 1
                self.block.RemoveParameterBlock( objPt )
                if self.solveOpts.linear_solver_ordering is not None:
                    self.solveOpts.linear_solver_ordering.Remove( objPt )
        del objPtIds2remove

        affectedCamIds = []
        for imgId in frozenset(affectedImgIds):
            image = self.images.get( imgId )
            if image is None:
                continue # image has already been popped by an outer call of this
            # as we continue upon encountering anything but TieImgObsData, it doesn't matter if we use image.rot or image.prc
            resBlocks = self.block.GetResidualBlocksForParameterBlock(image.rot)
            if len(resBlocks) >= self.minResBlocksPerPho:
                continue
            costDatas = [ getattr( self.block.GetCostFunctionForResidualBlock(resBlock), 'data', None )
                          for resBlock in resBlocks ]
            if any( not isinstance( costData, self.TieImgObsData ) for costData in costDatas ):
                continue
            assert all( imgId == costData.imgId for costData in costDatas )
            self.logger.warning(f'Remove pho {Path(image.path).name}: only {len(resBlocks)} tie image points and no other residual blocks')
            affectedCamIds.append( image.camId )
            # RemoveParameterBlock removes all residual blocks that depend on the parameter block.
            # However, we need to check if all object points of the removed image points are still observed in enough photos.
            # Thus, remove the image point residual blocks beforehand, and call RemoveParameterBlock afterwards.
            self.images.pop( imgId, None )
            nDeacs[:] += self._deac( list(resBlocks) )
            for par in ( image.prc, image.rot ):
                assert self.block.NumResidualBlocksForParameterBlock( par ) == 0
                self.block.RemoveParameterBlock( par )
                if self.solveOpts.linear_solver_ordering is not None:
                    self.solveOpts.linear_solver_ordering.Remove( par )

        # remove cameras with no images
        for camId in frozenset(affectedCamIds):
            if any( camId == image.camId for image in self.images.values()  ):
                continue
            camera = self.cameras.pop(camId, None)
            if camera is None:
                continue # camera has already been popped by an outer call of this
            self.logger.warning('Remove camera {}, as it has been left with no images',camId)
            for par in ( camera.ior, camera.adp ):
                assert self.block.NumResidualBlocksForParameterBlock( par ) == 0
                self.block.RemoveParameterBlock( par )
                if self.solveOpts.linear_solver_ordering is not None:
                    self.solveOpts.linear_solver_ordering.Remove( par )
        return nDeacs