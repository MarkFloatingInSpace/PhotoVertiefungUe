# -*- coding: cp1252 -*-
import os
import numpy as np

from oriental import config

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

def plotBlockWebGL( outFn, objPts, images, axisLen = .1, cameraPositionZ = 5., cameraFarPlane=5000 ):
    # mostly derived from three.js-r76\examples\webgl_buffergeometry_points.html
    with open( outFn, 'wt', encoding='UTF-8' ) as fout:
        fout.write( """<!DOCTYPE html>
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
""" )
        fout.write( """
	<body>
		<div id="container"></div>
		<div id="info">
			<a href="http://arap.univie.ac.at/" target="_blank">OrientAL</a> - Sparse Reconstruction ({} obj.pts., {} photos)</br>
			MOVE mouse &amp; press LEFT: rotate, MIDDLE: zoom, RIGHT: pan, +|-: double/halve point size, *|/: double/halve camera size
		</div>
""".format( objPts.shape[0], len(images) ) )

        if 0:
            fout.write("""
		<script src="file://{three}/build/three.js"></script>
		<script src="file://{three}/examples/js/controls/TrackballControls.js"></script>
		<script src="file://{three}/examples/js/WebGL.js"></script>
		<script src="file://{three}/examples/js/libs/stats.min.js"></script>
		<script src="file://{three}/examples/fonts/helvetiker_regular.typeface.js"></script>
""".format( three=config.threeJs.replace("\\","/") ) )
        else:
            fout.write( javascriptBegin() )
            with open( os.path.join( config.threeJs, "build/three.min.js" ), "rt", encoding='UTF-8' ) as fin:
                fout.write( fin.read() )
            fout.write( '\n\n' )
            # define global variable "font", to be used in addCamera
            fout.write( "var font = new THREE.Font(" )
            with open( os.path.join( config.threeJs, "examples/fonts/helvetiker_regular.typeface.json" ), "rt", encoding='UTF-8' ) as fin:
                fout.write( fin.read() )
            fout.write( ');\n\n' )

            with open( os.path.join( config.threeJs, "examples/js/libs/stats.min.js" ), "rt", encoding='UTF-8' ) as fin:
                fout.write( fin.read() )
            fout.write( '\n\n' )
            fout.write( javascriptEnd() )

        fout.write( javascriptBegin() )

        # With three-js 73, THREE.Points.sortParticles has been removed. Now it seems that Points are always drawn on top of the cameras, no matter if they are before or behind them.
        # As a work-around, setting the pyramid and text materials' property 'transparent', renders them after all non-transparent objects (including Points).
        # For the pyramid, use THREE.FlatShading, so the pyramid edges are not smoothed, and the camera orientations are well recognizable.
        fout.write( """
			function addPts( thing ) {
				var i = 0;
""" )

        for pt in objPts:
            pt_ = pt['X']
            rgb = pt['RGB']
            fout.write( "addPt(thing,i++,{},{},{},{:.3f},{:.3f},{:.3f});\n".format( pt_[0], pt_[1], pt_[2], *rgb ) )

        fout.write("""
			}

			function addCameras() {
				cameras = [
""" )
        for idx,img in enumerate(images):
            fout.write("addCamera( [{w},{h},{f}], [{X0}, {Y0}, {Z0}], [{ang1}, {ang2}, {ang3}, '{angParam}'], '{name}' ){comma}\n".format(
                w=img['whf'][0],
                h=img['whf'][1],
                f=img['whf'][2],
                X0=img['X0'] [0],
                Y0=img['X0'] [1],
                Z0=img['X0'] [2],
                ang1=img['angles'][0],
                ang2=img['angles'][1],
                ang3=img['angles'][2],
                angParam=img['rotParam'],
                name=img['name'],
                comma = ',' if idx < len(images)-1 else ""
                ) )

        fout.write("""
				];
			}
""")
        fout.write( """
			var axisLen = {};
""".format( axisLen ) )
        fout.write( """
			var webGL = WEBGL.isWebGLAvailable();
			var stats, camera, controls, scene, renderer, cameras, ptcl;
			var pyrMat = new THREE.MeshStandardMaterial( { color: 0xff00ff, transparent : true, flatShading : true } );
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
        fout.write("				camera = new THREE.PerspectiveCamera( 60 /*vertical FoV [deg]*/, window.innerWidth / window.innerHeight /*width-to-height aspect ratio*/ );\n" )
        fout.write("				camera.position.z = {};\n".format(cameraPositionZ) )
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

			function addCamera( whf, pos, rot, name ) {
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

        fout.write( "{}".format( objPts.shape[0] ) )

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


if __name__ == '__main__':
    import os, math

    # http://blogoben.wordpress.com/2011/06/05/webgl-basics-5-full-transformation-matrix/

    outFn = r"D:\arap\data\140217_relOri_Test_md\test2\2\relOri\block.html"

    objPts  = np.loadtxt( r"D:\arap\data\140217_relOri_Test_md\test2\2\relOri\objPts.txt",
                          dtype=[ ('X', np.float, 3), ('RGB', np.int, 3) ] )

    images  = np.loadtxt( r"D:\arap\data\140217_relOri_Test_md\test2\2\relOri\images.txt",
                          dtype=[ ('id', np.int), ('X0', np.float, (3,)), ('angles', np.float, (3,)), ('rotParam',(np.str,10)), ('path',(np.str,500)) ],
                          ndmin=1 )
    # paths are read including double-quotes at begin and end. substringRange removes them, too.
    substringRange = (  len(os.path.commonprefix([ fn.lower()       for fn in images['path'] ])),
                       -len(os.path.commonprefix([ fn[::-1].lower() for fn in images['path'] ])) )

    cameras = np.loadtxt( r"D:\arap\data\140217_relOri_Test_md\test2\2\relOri\cameras.txt",
                          dtype=[ ('id',np.int), ('ior',np.float,(3,)) ],
                          ndmin=1 )
    cams = { row['id']:row['ior'] for row in cameras }
    imgs = np.empty( images.shape[0], dtype=[ ( 'X0'      , np.float     , (3, ) ),
                                              ( 'angles'  , np.float     , (3, ) ),
                                              ( 'rotParam', (np.str,10) ),
                                              ( 'ior'     , np.float     , (3,)  ),
                                              ( 'name'    , (np.str,500)         ) ] )

    for idx in range(imgs.shape[0]):
        angles = images[idx]['angles'] / 200. * np.pi
        imgs[idx]['X0'][:] = images[idx]['X0']
        imgs[idx]['angles'][:] = angles
        rotParam = images[idx]['rotParam']
        if rotParam == 'omfika':
            imgs[idx]['rotParam'] = 'XYZ'
        else:
            # three.js has no option to rotation around the same axis twice.
            # If a rotation matrix is passed, then three.js decomposes it first = inefficient.
            # Better pass a quaternion instead.
            raise Exception( "Rotation matrix parameterization not supported: '{}'".format( rotParam ) )

        id = images[idx]['id']
        imgs[idx]['ior'][:] = cams[id]
        imgs[idx]['name'] = images[idx]['path'][ substringRange[0]:substringRange[1] ]

    plotBlockWebGL( outFn, objPts, imgs )
