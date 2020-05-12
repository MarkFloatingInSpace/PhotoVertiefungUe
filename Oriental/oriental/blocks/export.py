# -*- coding: cp1252 -*-
"export blocks to various formats"

from oriental import config, adjust, ori, log, utils
import oriental.utils.filePaths

from contracts import contract, new_contract
import numpy as np
from scipy import linalg, spatial

import colorsys, struct
from itertools import chain
from pathlib import Path

logger = log.Logger(__name__)

new_contract('Problem',adjust.Problem)

@contract
def webGL( fn : Path,
           block : adjust.Problem,
           cameras : dict,
           images : dict,
           objPts : dict,
           imgObsDataType : type,
           axisLen : 'float,>0|None' = None) -> None:
    "export image orientations and object points as WebGL via three.js, embedded in HTML"
    # mostly derived from three.js-r76\examples\webgl_buffergeometry_points.html

    _three = Path( config.threeJs )

    def javascriptBegin():
        return """
		    <script type='text/javascript'>
			    /* <![CDATA[ */
    """

    def javascriptEnd():
        return """
			    /* ]]> */
		    </script>
    """
    prcs = np.array([ img.prc for img in images.values() ])

    if axisLen is None:
        tree = spatial.cKDTree( prcs )
        dists, indxs = tree.query(prcs, k=2, n_jobs=-1)
        medInterPrcDist = np.median( dists[:,1] )
        axisLen = medInterPrcDist / 5

    cameraPositionZ = axisLen * 50
    cameraFarPlane= linalg.norm( np.fromiter( chain.from_iterable(objPts.values()), float).reshape((-1,3)).ptp(axis=0) ) * 2 # twice the diagonal of the bounding box

    # reduce by the coordinate-wise median, and scale uniformly in all directions by sigma MAD of X,Y-coordinates
    offset = np.median( prcs, axis=0 )
    sigma = 1.4826 * np.median( np.abs( prcs - offset )[:,:2] )
    scale = 1./sigma
    axisLen *= scale
    cameraPositionZ *= scale
    cameraFarPlane *= scale

    shortNames = utils.filePaths.ShortFileNames([img.path for img in images.values()])

    with fn.open('wt', encoding='UTF-8') as fout:
        fout.write("""<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
		<title>Sparse Reconstruction - OrientAL</title>
		<link rel='shortcut icon' href='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3gcGDRIHneMiEAAAAkhJREFUOMt9k09I02EYxz+vjszDMqYTtkMNYpIHabSDBy8RdPFkB/Egg+1ih0TwEITVwYO1DoGeFUeevIhFh64R4klR2DyMTSSc+222gyXYqu39dmg/Geb6wnN4eL7f5+/7GkmGC5DUA9wFrgEYYyqSMsaYykWu54J//+Tk5Pna2tq9dDpNsVgEIBQKEY1Gfw0PD3/2er0vgU/N1UzDXiwsLMjr9QrQyMiIlpeXlUqlNDo6KkBdXV2an5+XpFeuzk2QmJ2dFSBAS0tLP621zyT5rbV+SU9WV1d/eDweAZqZmZG19pGboHNra8txxZOTk5L0uKkz12LT09PnRTY3N79K6kTSg6mpqfNAJpPZv0RsJJlsNpt1eRMTE5L0sA2I7O7uAtDd3U1/f/8GLdDX17fh9/sB2NvbA7jdBlyp1WoA+Hw+jDHfaY1KMBgEoF6vA3jagGJvby8A5XIZSb7/JLheqfx9Co1OSki6ubKy8tudbXt7+8ha23HJDtqz2WzO5S0uLtYl3UKSOTs7ezs4OChAY2NjkvRGUnuzWNLreDwuQJFIRKenp+/dM45L+lAqlcpDQ0MCFI/HVSgUdiR1Wms7Dg8Pd2KxmAANDAxof3//m6SPksbJ5/PpcDgsx3G+1Go1ra+vKxQKaW5uTpKi1to7yWRSgUBAqVRK1WpVlUrlKBwOK5/Ppz0AuVyOQCBwo3lbjuMA9BpjasfHxziOQyKRIJFIAARbfaZzFAoFgB7AHhwctDyLaexg5J+AMVXgacNNSrp6CefdH+RQoakedXzxAAAAAElFTkSuQmCC' />
		<style>
			body {
				font-family:Monospace;
				font-size:13px;
				text-align:center;
				font-weight: bold;
				margin: 0px;
				overflow: hidden;
			}

			#info {
				color:#000;
				position: absolute;
				top: 0px;
				width: 100%;
				padding: 5px;
			}

			a {
				color: red;
			}
		</style>
	</head>
""")
        fout.write(f"""
	<body>
		<div id="container"></div>
		<div id="info">
			<a href="http://arap.univie.ac.at/" target="_blank">OrientAL</a> - Sparse Reconstruction ({len(objPts)} obj.pts., {len(images)} cameras)</br>
			MOVE mouse &amp; press LEFT: rotate, MIDDLE: zoom, RIGHT: pan, +|-: double/halve point size, *|/: double/halve camera size
		</div>
""")

        if 0:
            fout.write("""
		<script src="file://{three}/build/three.js"></script>
		<script src="file://{three}/examples/js/controls/TrackballControls.js"></script>
		<script src="file://{three}/examples/js/WebGL.js"></script>
		<script src="file://{three}/examples/js/libs/stats.min.js"></script>
		<script src="file://{three}/examples/fonts/helvetiker_regular.typeface.js"></script>
""".format( three= str(_three).replace('\\','/') ) )
        else:
            fout.write( javascriptBegin() )
            for path in ( _three / 'build' / 'three.min.js',
                          _three / 'examples' / 'js' / 'libs' / 'stats.min.js' ):
                with path.open( 'rt', encoding='UTF-8' ) as fin:
                    fout.write( fin.read() )
                    fout.write( '\n\n' )
            # define global variable "font", to be used in addCamera
            fout.write( "var font = new THREE.Font(" )
            with ( _three / 'examples' / 'fonts' / 'helvetiker_regular.typeface.json' ).open('rt', encoding='UTF-8') as fin:
                fout.write( fin.read() )
            fout.write(');\n\n')

            fout.write(javascriptEnd())

        fout.write(javascriptBegin())

        fout.write("""
			function addPts(thing) {
				var i = 0;
""")
        for objPt in objPts.values():
            rgbs = []
            for cost in ( block.GetCostFunctionForResidualBlock(resBlock) for resBlock in block.GetResidualBlocksForParameterBlock(objPt) ):
                costData = getattr( cost, 'data', None )
                if isinstance( costData, imgObsDataType ):
                    rgbs.append( costData.rgb )
            if len(rgbs):
                rgb = np.median( np.array(rgbs), axis=0 )
            else:
                rgb = np.zeros(3)
            pt = ( objPt - offset ) * scale
            fout.write(f"addPt(thing,i++,{pt[0]},{pt[1]},{pt[2]},{rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f});\n")

        fout.write("""
			}

			function addCameras() {
				cameras = [""")

        camId2Idx = {camId : idx for idx, camId in enumerate(cameras)}
        for img in images.values():
            camera = cameras[img.camId]
            focal_px = camera.ior[2] / img.pix2cam.meanScaleForward()
            ior = np.array([img.nCols, img.nRows, focal_px]) * scale
            prc = ( img.prc - offset ) * scale
            # while three-js supports all permutations of rotations about the 3 axes, it does not support 3 rotations about only 2 axes, so we force omfika here.
            ofk = ori.omfika( ori.euler2matrix(img.rot) ) / 200. * np.pi
            fout.write(f"""
addCamera( [{ior[0]},{ior[1]},{ior[2]}],[{prc[0]},{prc[1]},{prc[2]}],[{ofk[0]},{ofk[1]},{ofk[2]},'XYZ'],'{shortNames(img.path)}', pyrMats[{camId2Idx[camera.id]}] ),""")

        fout.write("""
				];
			}
""")
        fout.write(f"""
			var axisLen = {axisLen};
""")
        # With three-js 73, THREE.Points.sortParticles has been removed. Now it seems that Points are always drawn on top of the cameras, no matter if they are before or behind them.
        # As a work-around, setting the pyramid and text materials' property 'transparent', renders them after all non-transparent objects (including Points).
        # For the pyramid, use flatShading, so the pyramid edges are not smoothed, and the camera orientations are well recognizable.
        # Display each group of images that were taken with the same camera in a different color.
        fout.write("""
			var pyrMats = [""")
        for idx, camId in enumerate(cameras):
            hue = 0.83333333 + idx / len(cameras)
            hue = hue % 1
            rgb = colorsys.hsv_to_rgb( hue, 1., 1. )
            hexRgb = '0x' + ''.join(f'{int(el*255):02X}' for el in rgb )
            fout.write(f"""
				new THREE.MeshStandardMaterial( {{ color: {hexRgb}, transparent : true, flatShading : true }} ),""")

        fout.write("""
			];""")

        fout.write("""

			var webGL = WEBGL.isWebGLAvailable();
			var stats, camera, controls, scene, renderer, cameras, ptcl;
			var txtMat = new THREE.MeshBasicMaterial( { color: 0xffffff, transparent : true } );

			init();
			animate();

			function init() {
				if ( ! webGL ) {
					var webGlErrorMessage = WEBGL.getWebGLErrorMessage();
					webGlErrorMessage.style.margin='auto';
					webGlErrorMessage.style.padding='inherit';
					webGlErrorMessage.style.width='auto';
					webGlErrorMessage.style.color='red';
					webGlErrorMessage.style.background='inherit';
					document.getElementById('info').appendChild(webGlErrorMessage);
				}

""")
# near- and far-plane may cause problems (objects invisible, even though in FoV. What might they help, actually? Let's not specify them.
#        fout.write("""
#				camera = new THREE.PerspectiveCamera( 60 /*vertical FoV [deg]*/, window.innerWidth / window.innerHeight /*width-to-height aspect ratio*/, .001 /*near plane*/, {farPlane} /*far plane*/ );
#				camera.position.z = {cameraPositionZ};""".format(farPlane=cameraFarPlane,cameraPositionZ=cameraPositionZ) )
        fout.write(f"""
				camera = new THREE.PerspectiveCamera( 60 /*vertical FoV [deg]*/, window.innerWidth / window.innerHeight /*width-to-height aspect ratio*/ );
				camera.position.z = {cameraPositionZ};\n""" )
        fout.write( """		

				controls = new THREE.TrackballControls( camera );
				controls.rotateSpeed = 1.0;
				controls.zoomSpeed = 1.2;
				controls.panSpeed = 0.8;
				controls.noZoom = false;
				controls.noPan = false;
				controls.staticMoving = true;
				controls.dynamicDampingFactor = 0.3;
				controls.keys = [ 65, 83, 68 ];
				controls.addEventListener( 'change', render );

				scene = new THREE.Scene();

				addPtcl();
				addCameras();

				var light = new THREE.AmbientLight( 0x222222 );
				scene.add( light );
				light = new THREE.HemisphereLight( 0xffffff, 0x222222, 0.6 );
				light.position.set( 0, 0, 100 );
				scene.add( light );

				if( webGL )
					renderer = new THREE.WebGLRenderer( { antialias: true } );
				else
					renderer = new THREE.SoftwareRenderer();
				renderer.setSize( window.innerWidth, window.innerHeight );
				renderer.setClearColor( 0x555555, 1 /*alpha*/ );

				var container = document.getElementById( 'container' );
				container.appendChild( renderer.domElement );

				stats = new Stats();
				container.appendChild( stats.domElement );

				document.addEventListener("keypress", onDocumentKeyPress, false);

				window.addEventListener( 'resize', onWindowResize, false );
			}

			function onWindowResize() {
				camera.aspect = window.innerWidth / window.innerHeight;
				camera.updateProjectionMatrix();

				renderer.setSize( window.innerWidth, window.innerHeight );

				controls.handleResize();

				render();
			}

			function animate() {
				requestAnimationFrame( animate );
				render();
				controls.update();
			}

			function render() {
				renderer.render( scene, camera );
				stats.update();
			}

			function onDocumentKeyPress(evt) {
				// Increase/decrease point size
				evt = evt || window.event;
				var charCode = (typeof evt.which == "number") ? evt.which : evt.keyCode;
				if( charCode ) {
					var str = String.fromCharCode(charCode);
					// Increase/decrease point size
					if( str == '+' ) {
						if( webGL )
							ptcl.material.size *= 2.;
						else
							for( var i=0; i < ptcl.length; i++ )
								ptcl[i].scale.multiplyScalar( 2. );
						render();
					}
					if( str == '-' ) {
						if( webGL )
							ptcl.material.size /= 2.;
						else
							for( var i=0; i < ptcl.length; i++ )
								ptcl[i].scale.divideScalar( 2. );
						render();
					}
					if( str == '*' ) {
						for( var i = 0; i < cameras.length; i++ )
							cameras[i].scale.multiplyScalar( 2.);
						render();
					}
					if( str == '/' ) {
						for( var i = 0; i < cameras.length; i++ )
							cameras[i].scale.divideScalar( 2. );
						render();
					}
				}
			}

			function addCamera( whf, pos, rot, name, pyrMat ) {
				var axes = new THREE.AxesHelper( axisLen );
				axes.rotation.fromArray( rot );
				axes.position.fromArray( pos );

				var text3d = new THREE.TextGeometry(
					name, {
						size: axisLen,
						height: axisLen/100.,
						curveSegments: 4,
						font: font // instance of THREE.Font defined globally above
					});
				text3d.computeBoundingBox();

				var textMesh = new THREE.Mesh( text3d, txtMat );
				textMesh.position.x = -0.5 * ( text3d.boundingBox.max.x - text3d.boundingBox.min.x );
				textMesh.position.y = -0.5 * ( text3d.boundingBox.max.y - text3d.boundingBox.min.y );
				textMesh.position.z = axisLen;
				axes.add( textMesh );

				var pyramid = new THREE.CylinderBufferGeometry( 0 /*radiusTop*/, .5 /*radiusBottom*/, 1 /*height*/, 4 /*radialSegments*/, 1 /*heightSegments*/, false /*openEnded*/ );

				var pyramidMesh = new THREE.Mesh( pyramid, pyrMat );
				pyramidMesh.rotation.y = Math.PI / 4.;
				pyramidMesh.rotation.x = Math.PI / 2.;

				var pyramidParent = new THREE.Object3D();
				pyramidParent.scale.x = 2 * axisLen;
				pyramidParent.scale.y = 2 * axisLen * whf[1] / whf[0];
				pyramidParent.scale.z = 2 * axisLen * whf[2] / whf[0];
				pyramidParent.position.z = -pyramidParent.scale.z / 2.;
				pyramidParent.add(pyramidMesh);
				axes.add( pyramidParent );

				scene.add(axes);
				return axes;
			}

			function addPt( thing, idx, x, y, z, r, g, b ) {
				if( webGL ) {
					var positions = thing[0];
					var colors = thing[1];
					positions[ idx*3 ]		 = x;
					positions[ idx*3 + 1 ] = y;
					positions[ idx*3 + 2 ] = z;
					colors[ idx*3 ]		 = r;
					colors[ idx*3 + 1 ] = g;
					colors[ idx*3 + 2 ] = b;
				} else {
					thing[idx].position.set(x,y,z);
					thing[idx].material.color.setRGB( r, g, b );
					scene.add(thing[idx]);
				}
			}

			function addPtcl() {
				var nPts = """ )

        fout.write( f"{len(objPts)}" )

        fout.write( """;
				if( webGL ) {
					var positions = new Float32Array( nPts * 3 );
					var colors = new Float32Array( nPts * 3 );
					var ptclArrays = new Array( positions, colors );
				} else {
					ptcl = new Array(nPts);
					for( var i = 0; i < nPts; i++ )
						ptcl[i] = new THREE.Sprite();
				}
                addPts(webGL ? ptclArrays : ptcl);
				if( webGL ) {
					var ptclGeometry = new THREE.BufferGeometry();
					ptclGeometry.addAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
					ptclGeometry.addAttribute( 'color', new THREE.BufferAttribute( colors, 3 ) );
					ptclGeometry.computeBoundingSphere();
					var ptclMaterial = new THREE.PointsMaterial( { size: .1, vertexColors: THREE.VertexColors, sizeAttenuation : true } );
					ptcl = new THREE.Points( ptclGeometry, ptclMaterial );
					scene.add( ptcl );
				}
			}
""" )
        fout.write( javascriptEnd() )
        fout.write( """

	</body>
</html>""" )

    logger.info(f'Reconstruction exported to {fn}')

@contract
def ply( fn : Path,
         cameras : dict,
         images : dict,
         objPts : dict,
         block : 'Problem|None', # pass None if problemOpts.enable_fast_removal==False - or otherwise, this will be very slow
         otherObjPtsList : 'seq(dict)' = [{}],
         focalLenInObjSpace : 'number,>0|None' = None, # automatic scaling fails e.g. for multi-camera platforms
         binary : bool = True ) -> None:
    """Export image orientations and object points in Stanford PLY format.
    Loadable with CloudCompare, etc.
    """
    if not focalLenInObjSpace:
        # scale the camera pyramid s.t. the focal length is as long as 1/3 of the median distance between each PRC and its nearest neighbor.
        prcs = np.array([ img.prc for img in images.values() ])
        tree = spatial.cKDTree(prcs)
        dists, indxs = tree.query(prcs, k=2, n_jobs=-1)
        medInterPrcDist = np.median( dists[:,1] )
        focalLenInObjSpace = medInterPrcDist / 3.

    nPts = len(objPts) + sum( len(el) for el in otherObjPtsList )
    # .ply binary
    with fn.open( 'wb' if binary else 'w' ) as fout:
        header = """ply
format {fmt} 1.0
comment generated by OrientAL
element vertex {nVertices}
property float64 x
property float64 y
property float64 z
property uint8 red
property uint8 green
property uint8 blue
property uint8 nImgPts
property uint8 type
element face {nFaces}
property list uint8 uint32 vertex_indices
end_header
""".format( fmt='binary_little_endian' if binary else 'ascii',
            nVertices=nPts + len(images)*(5+8*3),
            nFaces=len(images)*(6+6*2*3) )
        if binary:
            header = header.encode('ascii')
        fout.write( header )
        if binary:
            format = struct.Struct('<dddBBBBB').pack
        else:
            format = lambda *args: '\t'.join( ['{}']*len(args) ).format(*args) + '\n'
        for objPt in objPts.values():
            rgb = np.ones(3) * .5
            nViews = 1
            if block is not None:
                resBlocks = block.GetResidualBlocksForParameterBlock(objPt)
                nViews = len(resBlocks)
                firstCost = block.GetCostFunctionForResidualBlock(resBlocks[0])
                if hasattr(firstCost,'data') and hasattr(firstCost.data,'rgb'):
                    rgb = np.mean([block.GetCostFunctionForResidualBlock(resBlock).data.rgb for resBlock in resBlocks], axis=0)
            fout.write( format( *chain( objPt, (rgb*255).astype(np.uint8), [nViews], [0] ) )  )
        for idx, otherObjPts in enumerate(otherObjPtsList,1):
            for objPt in otherObjPts.values():
                nViews = 1
                if block is not None:
                    nViews = block.NumResidualBlocksForParameterBlock(objPt)
                fout.write( format( *chain( objPt, np.array([255,255,255],np.uint8), [nViews], [idx] ) )  )

        camId2rgb = {}
        for idx, camId in enumerate(cameras):
            hue = 0.83333333 + idx / len(cameras)
            hue = hue % 1
            rgb = colorsys.hsv_to_rgb( hue, 1., 1. )
            camId2rgb[camId] = ( np.array(rgb)*255 ).astype(np.uint8)

        for img in images.values():
            camera = cameras[img.camId]
            focal_px = camera.ior[2]
            if hasattr(img, 'pix2cam'):
                focal_px /= img.pix2cam.meanScaleForward()
            scale = focalLenInObjSpace / focal_px
            pts = np.array([ [            0,            0,              0 ],
                             [  img.nCols/2,  img.nRows/2, -focal_px ],
                             [ -img.nCols/2,  img.nRows/2, -focal_px ],
                             [ -img.nCols/2, -img.nRows/2, -focal_px ],
                             [  img.nCols/2, -img.nRows/2, -focal_px ] ], float)
            # scale pyramids
            pts *= scale
            R = ori.euler2matrix( img.rot )
            pts = R.dot( pts.T ).T + img.prc
            for pt in pts:
                fout.write( format( *chain( pt, camId2rgb[img.camId], [0], [0] ) ) )

            # axes as cuboids
            axLen = ( img.nCols + img.nRows ) / 4.
            axWid = axLen / 20.
            pts = np.array([ [ 0.,  axWid,  axWid ],
                             [ 0., -axWid,  axWid ],
                             [ 0., -axWid, -axWid ],
                             [ 0.,  axWid, -axWid ] ] )
            pts = np.r_[ pts, pts + np.array([ axLen, 0., 0. ]) ] * scale
            for iAx in range(3):
                if iAx==0:
                    R2 = np.eye(3)
                    col = np.array([255,0,0],np.uint8)
                elif iAx==1:
                    R2 = np.array([[  0, 1, 0 ],
                                    [ -1, 0, 0 ],
                                    [  0, 0, 1 ]],float).T
                    col = np.array([0,255,0],np.uint8)
                else:
                    R2 = np.array([[  0, 0, 1 ],
                                    [  0, 1, 0 ],
                                    [ -1, 0, 0 ]],float).T
                    col = np.array([0,0,255],np.uint8)
                pts_ = R.dot(R2).dot( pts.T ).T + img.prc
                for pt in pts_:
                    fout.write( format( *chain( pt, col, [0], [0] ) ) )

        if binary:
            format = struct.Struct('<BIII').pack

        for iImg in range(len(images)):
            # CloudCompare does not support:
            # - reading polylines from PLY files
            # - faces with a vertex count other than 3                          
            offset = nPts + iImg * (5+8*3)
            for iVtxs in np.array([[0, 1, 2],
                                   [0, 2, 3],
                                   [0, 3, 4],
                                   [0, 4, 1],
                                   [1, 2, 3],
                                   [3, 4, 1]], int):
                fout.write( format( *chain( [3], iVtxs+offset ) ) )
            for iAx in range(3):
                offset = nPts + iImg * (5+8*3) + 5 + 8*iAx
                for iVtxs in np.array([[0, 3, 2],
                                       [2, 1, 0],
                                       [0, 1, 5],
                                       [5, 4, 0],
                                       [0, 4, 7],
                                       [7, 3, 0],
                                       [2, 3, 7],
                                       [7, 6, 2],
                                       [1, 2, 6],
                                       [6, 5, 1],
                                       [4, 5, 6],
                                       [6, 7, 4]], int):
                    fout.write( format( *chain( [3], iVtxs+offset ) ) )

    logger.info(f'Reconstruction exported to {fn}')
