# -*- coding: cp1252 -*-
import environment
import oriental
from oriental import config, blocks
import oriental.blocks.types

import numpy as np

import unittest
import pickle

class TestBlocksTypes(unittest.TestCase):
    def test_Compact(self):
        cam0 = blocks.types.Camera( ior=np.arange(3,dtype=float), adp=np.arange(9,dtype=float) )
        Camera = blocks.types.createCompact( blocks.types.Camera, ['id'] )
        cam = Camera( ior=np.arange(3,dtype=float), adp=np.arange(9,dtype=float), id=0 )
        self.assertIsInstance( cam, blocks.types.Camera )
        self.assertIsInstance( cam, Camera )
        self.assertFalse( hasattr(cam,'__dict__') )
        cam2 = Camera( ior=np.arange(3,dtype=float), adp=np.arange(9,dtype=float), id=1 )
        s = pickle.dumps((cam,cam2),pickle.HIGHEST_PROTOCOL)
        cpy,cpy2 = pickle.loads(s)
        self.assertEqual( cpy.id, cam.id )
        self.assertIs( cpy.__class__, cpy2.__class__ )
        self.assertIs( cam.__class__, cpy.__class__ )

    def test_CompactDerived(self):
        Camera = blocks.types.createCompact( blocks.types.Camera, ['id'] )
        CameraDerived = blocks.types.createCompact( Camera, ['keypts'] )
        cam = CameraDerived( ior=np.arange(3,dtype=float), adp=np.arange(9,dtype=float), id=1, keypts='missing, dude' )
        self.assertIsInstance( cam, blocks.types.Camera )
        self.assertIsInstance( cam, Camera )
        self.assertIsInstance( cam, CameraDerived )
        self.assertFalse( hasattr(cam,'__dict__') )
        s = pickle.dumps(cam,pickle.HIGHEST_PROTOCOL)
        cpy = pickle.loads(s)
        self.assertEqual( cpy.id, cam.id )
        self.assertEqual( cpy.keypts, cam.keypts )
        self.assertIs( cpy.__class__, cam.__class__ )
        self.assertIsInstance( cpy, blocks.types.Camera )
        self.assertIsInstance( cpy, Camera )
        self.assertIsInstance( cpy, CameraDerived )
        self.assertFalse( hasattr(cpy,'__dict__') )

if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestBlocksTypes.test_CompactDerived',
                       exit=False )
