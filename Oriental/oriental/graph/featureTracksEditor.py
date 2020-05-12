# -*- coding: cp1252 -*-
from oriental import graph, log, utils
import oriental.utils.stats

import collections
import collections.abc

from contracts import contract
import numpy as np

logger = log.Logger(__name__)

class FeatureTracksEditor:
    @contract
    def __init__( self,
                  edge2matches : dict, # { (edgeId1,edgeId2) : array[Nx2](int) }; array contains integers that are unique for each image (indices into each image's array of keypoints)
                  images : dict # { imgId : Image }
                ):
        self.edge2matches = edge2matches
        self.images = images
        self.buildFeatureTracks()

    def buildFeatureTracks( self ):
        "build featureTracks using edge2matches"
        self.featureTracks = graph.ImageFeatureTracks()
        for edge, matches in self.edge2matches.items():
            for match in np.atleast_2d( matches ):
                self.featureTracks.join( graph.ImageFeatureID( edge[0], match[0].item() ),
                                         graph.ImageFeatureID( edge[1], match[1].item() ) )
        self.featureTracks.compute()

    def _dropMultipleProjectionMatches( self ):
        """remove from self.edge2matches all matches that are multiple projections of the same object point according to featureTracks"""
        # Jeder track darf pro pho nur einmal beobachtet sein. Ansonsten beide (alle) matches entfernen, aber nur in diesem einen Bild! 
        # for each image pair, collect a set of match indices that are deemed inconsistent, and thus shall be removed
        Msg = collections.namedtuple( 'Msg', ( 'iImg1', 'iImg2', 'nRemoved', 'nRemain' ) )  
        edge2matchesToRemove = {}
        for imgId in self.images:
            # map: featureTrack -> first encountered corresponding (edge,iMatch) for current image
            featureTrack2FirstEdgeMatch = {}
            for thisEdge, matches in self.edge2matches.items():
                if imgId not in thisEdge:
                    continue
                iImgInThisEdge = thisEdge.index(imgId)
                assert thisEdge[iImgInThisEdge] == imgId
                for iThisMatch in range(matches.shape[0]):
                    iThisKeypt = matches[iThisMatch,iImgInThisEdge].item()
                    featureTrack = self.featureTracks.component( graph.ImageFeatureID( imgId, iThisKeypt ) )[1]
                    thatEdge, iThatMatch = featureTrack2FirstEdgeMatch.setdefault( featureTrack, (thisEdge, iThisMatch) )
                    if (thatEdge, iThatMatch) == (thisEdge, iThisMatch):
                        continue
                    iImgInThatEdge = thatEdge.index(imgId)
                    iThatKeypt = self.edge2matches[thatEdge][iThatMatch,iImgInThatEdge]
                    if iThatKeypt == iThisKeypt:
                        # the current and the first match correspond to the same track.
                        continue
                    # if they refer to different feature points, then remove both matches!
                    for edge,iMatch in [ (thisEdge,iThisMatch),
                                         (thatEdge,iThatMatch) ]:
                        assert imgId in edge
                        edge2matchesToRemove.setdefault( edge, set() ).add( iMatch )
                        
        del featureTrack2FirstEdgeMatch        
        
        logger.info('Remove multiple projections')
        msgs = []
        removedEdges = []
        for edge, matches in edge2matchesToRemove.items():
            matchesOrig = self.edge2matches[edge]
            nRemaining = matchesOrig.shape[0] - len(matches)
            if nRemaining >= 5:
                keep = np.ones( matchesOrig.shape[0], dtype=np.bool )
                keep[ np.array( list(matches) ) ] = False
                #self.edge2matches[edge].resize( (self.edge2matches[edge].shape[0] - len(matches), 2 ) ) # resize in-place; error: array does not own its data!
                #self.edge2matches[edge][:] = self.edge2matches[edge][keep,:] # copy in-place!
                self.edge2matches[edge] = matchesOrig[keep,:]
            else:
                # 2014-05-21: avoid edges that do not contain any matches (important), avoid edges that contain less matches than the minimum needed for rel. ori. (for performance)
                del self.edge2matches[edge]
                removedEdges.append( edge )
                nRemaining = 0

            msgs.append( Msg( edge[0], edge[1], len(matches), nRemaining ) )

        return msgs, removedEdges

    @contract
    def dropMultipleProjectionTracks( self ) -> None:
        """drop tracks and matches from featureTracks and edge2matches that are multiple projections of an object point into the same image"""
        logger.info('Detect multiple projections') 
        msgs,_ = self._dropMultipleProjectionMatches()
        if msgs:
            # efficiently compute min, median, max:
            nRemovedPercRemovedRemain = np.array([ ( msg.nRemoved, msg.nRemoved / (msg.nRemoved+msg.nRemain), msg.nRemain )
                                                   for msg in msgs ])

            statistics = 'Removed multiple projections statistics\n' \
                         'statistic\t#removed\t%removed\t#remaining\n' \
                         'min\t{:4.0f}\t{:2.0%}\t{:4.0f}\n' \
                         'median\t{:4.0f}\t{:2.0%}\t{:4.0f}\n' \
                         'max\t{:4.0f}\t{:2.0%}\t{:4.0f}\n'.format( *utils.stats.minMedMax( nRemovedPercRemovedRemain ).flat )
                         
            logger.log( log.Severity.info,
                        log.Sink.all if len(msgs) < 500 else log.Sink.file,
                        'Removed multiple projections\n'
                        'pho1\tpho2\t#removed\t%removed\t#remaining\n'
                        '{}\v'
                        '{}',
                        '\n'.join( ( "{}\t{}\t{:4}\t{:2.0%}\t{:4}".format( 
                                    self.images[msg.iImg1].path,
                                    self.images[msg.iImg2].path,
                                    msg.nRemoved,
                                    msg.nRemoved / (msg.nRemoved+msg.nRemain),
                                    msg.nRemain ) for msg in msgs ) ),
                        statistics )
            if not len(msgs) < 500:
                logger.infoScreen( statistics )

            logger.info('Re-compute feature tracks')
            self.buildFeatureTracks()

    @contract
    def thinOutTracks( self,
                       nCols : int,
                       nRows : int,
                       minFeaturesPerCell : int ) -> list:
        """Reduce the number of feature tracks, favouring tracks with high multiplicity and ensuring that all images remain orientable.
        
        Divide the area of each image into a grid of resolution `nCols` x `nRows`.
        Remove as many feature tracks as possible, such that in each cell of each image, at least `minFeaturesPerCell` image points remain.
        Try to keep tracks with large multiplicities, discard tracks with low ones."""
        logger.info( 'Thin out feature tracks on {}x{} (cols x rows) grids, keeping at least {} features per cell with largest multiplicity', nCols, nRows, minFeaturesPerCell )

        def tracks2Keep():
            def getNRowsNCols( image ):
                if image.nCols < image.nRows:
                    return nCols, nRows
                return nRows, nCols 

            def getRowCol( feature ):
                img = self.images[feature.iImage]
                pt = img.keypts[feature.iFeature]
                nrows, ncols = getNRowsNCols( img )
                col = int(  pt[0] / ( img.nCols / ncols ) )
                row = int( -pt[1] / ( img.nRows / nrows ) )
                return row, col

            keptTracks = []
            featureCounts = { imgId : np.zeros( getNRowsNCols(image), int ) for imgId,image in self.images.items() }
            nCellsLeft = len(featureCounts) * nRows * nCols
            components = list( self.featureTracks.components().items() )
            components.sort( key=lambda x: len(x[1]), reverse=True )
            for component, features in components:
                for feature in features:
                    row, col = getRowCol(feature)
                    if featureCounts[feature.iImage][row,col] < minFeaturesPerCell:
                        break
                else:
                    continue
                for feature in features:
                    row, col = getRowCol(feature)
                    featureCount = featureCounts[feature.iImage]
                    featureCount[row,col] += 1
                    if featureCount[row,col] == minFeaturesPerCell:
                        nCellsLeft -= 1
                keptTracks.append(component)
                if not nCellsLeft: # it is quite unlikely that all cells of all images get filled. However, this test doesn't cost much, so let's test and break early in that case.
                    break
            logger.log( log.Severity.info,
                        log.Sink.all if len(featureCounts) < 500 else log.Sink.file,
                        'Thinned out features per image\n'
                        'img\t#features\t#fullCells\n'
                        '{}',
                        '\n'.join( '{}\t{}\t{}'.format( self.images[imgId].path, counts.sum(), np.sum( counts >= minFeaturesPerCell ) )
                                   for imgId,counts in featureCounts.items() ) )
            return keptTracks

        keptTracks = frozenset(tracks2Keep())
        removedEdges = []
        edge2matches_old = self.edge2matches.copy()
        self.edge2matches.clear()
        for edge,oldMatches in edge2matches_old.items():
            matches = []
            for oldMatch in oldMatches:
                found, component = self.featureTracks.component( graph.ImageFeatureID( edge[0], oldMatch[0].item() ) )
                assert found
                if component in keptTracks:
                    matches.append( oldMatch )
            if len(matches):
                self.edge2matches[edge] = np.array(matches, oldMatches.dtype)
            else:
                removedEdges.append(edge)

        nTracksBefore = self.featureTracks.nComponents()
        logger.info('Re-compute feature tracks')
        self.buildFeatureTracks()
        nTracksAfter = self.featureTracks.nComponents()
        logger.info('thinout: #objPts reduced to {}({:2.0%}) from {}', nTracksAfter, nTracksAfter/nTracksBefore, nTracksBefore )
        return removedEdges