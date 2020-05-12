# -*- coding: cp1252 -*-
import environment
from oriental import config, graph

import unittest
import numpy as np

class TestGraph(unittest.TestCase):    

    def test_ImageFeatureTracks(self):
        tracks = graph.ImageFeatureTracks()
        
        feat1 = graph.ImageFeatureID( 2, 3 )
        feat2 = graph.ImageFeatureID( 3, 4 )
        feat3 = graph.ImageFeatureID( 4, 5 )
        feat4 = graph.ImageFeatureID( 3, 6 )
        feat5 = graph.ImageFeatureID( 5, 0 )
        
        tracks.join( feat1, feat3 )
        tracks.join( feat2, feat4 )
        tracks.join( feat4, feat5 )
        
        tracks.join( feat4, feat5 )
        tracks.join( feat4, feat4 )
        
        nComps = tracks.compute()
        self.assertEqual( nComps, 2 )
        
        self.assertTrue( tracks.component( feat1 )[0] )
        self.assertEqual( tracks.component( feat1 )[1], 0 )
        self.assertTrue( tracks.component( feat2 )[0] )
        self.assertEqual( tracks.component( feat2 )[1], 1 )
        self.assertTrue( tracks.component( feat3 )[0] )
        self.assertEqual( tracks.component( feat3 )[1], 0 )
        self.assertTrue( tracks.component( feat5 )[0] )
        self.assertEqual( tracks.component( feat5 )[1], 1 )
        self.assertEqual( tracks.component( graph.ImageFeatureID( 10, 13 ) )[0], False )
        
    def test_ImageConnectivity(self):
        img0 = graph.ImageConnectivity.Image(0)
        img1 = graph.ImageConnectivity.Image(1)
        img2 = graph.ImageConnectivity.Image(2)
        
        conn = graph.ImageConnectivity()
        conn.addEdge( graph.ImageConnectivity.Edge( img0, img1, quality=1. ) )
        conn.addEdge( graph.ImageConnectivity.Edge( img0, img2, quality=10. ) )
        conn.addEdge( graph.ImageConnectivity.Edge( img1, img2, quality=2. ) )
        
        self.assertEqual( conn.nVertices(), 3 )
        self.assertEqual( conn.nConnectedComponents(), 1 )
        self.assertEqual( conn.nEdges(), 3 )

        imgPair = graph.ImageConnectivity.Edge()
        self.assertTrue( conn.nextBestEdge( imgPair ) )
        self.assertEqual( imgPair.img1.idx, 0 )
        self.assertEqual( imgPair.img2.idx, 2 )
        
        imgPair.img1.state = graph.ImageConnectivity.Image.State.oriented
        imgPair.img2.state = graph.ImageConnectivity.Image.State.oriented
        conn.setImageState( imgPair.img1 )
        conn.setImageState( imgPair.img2 )
        imgPair.state = graph.ImageConnectivity.Edge.State.used
        conn.setEdgeState( imgPair )
        
        self.assertEqual( [ img.idx for img in conn.orientedImages() ], [0, 2] )
        self.assertEqual( [ img.idx for img in conn.unorientedImages() ], [1] )

        unoriented = conn.unorientedImagesAdjacent2Oriented()
        self.assertEqual( [ img.idx for img in unoriented ], [1] )

        minCut = conn.minCut(True)
        np.testing.assert_array_equal( minCut.idxsImagesSmallerSet, np.array([1]))
        dummy=0

    def test_ImageConnectivityImageHashable(self):
        im1 = graph.ImageConnectivity.Image(1)
        im2 = graph.ImageConnectivity.Image(1)
        self.assertTrue( im1 in [im2] )
        
if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestGraph.test_ImageConnectivity',
                       exit=False )
