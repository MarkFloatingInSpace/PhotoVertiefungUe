import itertools, struct, shutil
from pathlib import Path
from sqlite3 import dbapi2
from collections import namedtuple

import numpy as np

from oriental import config, adjust, blocks, ori, utils
import oriental.adjust.cost
import oriental.adjust.loss
import oriental.adjust.local_parameterization
from oriental.adjust import parameters
import oriental.utils.db
import oriental.utils.filePaths

ArrayWithId = namedtuple( 'ArrayWithId', 'id arr' )
#import oriental.blocks.db

#cameras, images, objPts, imgObs = blocks.db.restore( Path(r'D:\AutoPlan3D\OrientAL\12Styropor_3x4\relOri.sqlite') )

def webGL( fn : Path,
           block : adjust.Problem,
           cameras : dict,
           images : dict,
           objPts : dict,
           imgObsDataType : type ) -> None:
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

    tree = spatial.cKDTree( prcs )
    dists, indxs = tree.query(prcs, k=2, n_jobs=-1)
    medInterPrcDist = np.median( dists[:,1] )

    axisLen = medInterPrcDist / 5
    cameraPositionZ = medInterPrcDist * 10
    cameraFarPlane= linalg.norm( np.fromiter( chain.from_iterable(objPts.values()), float).reshape((-1,3)).ptp(axis=0) ) * 2 # twice the diagonal of the bounding box

    # reduce by the coordinate-wise median, and scale uniformly in all directions by sigma MAD of X,Y-coordinates
    offset = np.median( prcs, axis=0 )
    sigma = 1.4826 * np.median( np.abs( prcs - offset )[:,:2] )
    scale = 1./sigma
    axisLen *= scale
    cameraPositionZ *= scale
    cameraFarPlane *= scale

    shortNames = utils.filePaths.ShortFileNames([ img.path for img in images.values() ])

    with fn.open( 'wt', encoding='UTF-8' ) as fout:
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
			<a href="http://arap.univie.ac.at/" target="_blank">OrientAL</a> - Sparse Reconstruction ({} obj.pts., {} cameras)</br>
			MOVE mouse &amp; press LEFT: rotate, MIDDLE: zoom, RIGHT: pan, +|-: double/halve point size, *|/: double/halve camera size
		</div>
""".format( len(objPts), len(images) ) )

        fout.write( javascriptBegin() )
        for path in ( _three / 'build' / 'three.min.js',
                        _three / 'examples' / 'js' / 'libs' / 'stats.min.js' ):
            with path.open( 'rt', encoding='UTF-8' ) as fin:
                fout.write( fin.read() )
                fout.write( '\n\n' )
        # define global variable "font", to be used in addCamera
        fout.write( "var font = new THREE.Font(" )
        with ( _three / 'examples' / 'fonts' / 'helvetiker_regular.typeface.json' ).open( 'rt', encoding='UTF-8' ) as fin:
            fout.write( fin.read() )
        fout.write( ');\n\n' )

        fout.write( javascriptEnd() )

        fout.write( javascriptBegin() )

        fout.write( """
			function addPts(thing) {
				var i = 0;
""" )
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
            fout.write( "addPt(thing,i++,{},{},{},{:.3f},{:.3f},{:.3f});\n".format( *chain( ( objPt - offset ) * scale, rgb ) ) )
        fout.write( """
			}

			function addCameras() {
				cameras = [
""" )

        fout.write(
            ',\n'.join( "addCamera( [{},{},{}], [{}, {}, {}], [{}, {}, {}, 'XYZ'], '{}' )".format(
                          *chain( np.array([img.nCols, img.nRows, cameras[img.camId].ior[2]]) * scale,
                                  ( img.prc - offset ) * scale,
                                  ori.omfika( ori.euler2matrix(img.rot) ) / 200. * np.pi, # while three-js supports all permutations of rotations about the 3 axes, it does not support 3 rotations about only 2 axes, so we force omfika here.
                                  [shortNames(img.path)] )
                        ) for img in images.values() )  )

        fout.write("""
				];
			}
""")
        fout.write( """
			var axisLen = {};
""".format( axisLen ) )
        # With three-js 73, THREE.Points.sortParticles has been removed. Now it seems that Points are always drawn on top of the cameras, no matter if they are before or behind them.
        # As a work-around, setting the pyramid and text materials' property 'transparent', renders them after all non-transparent objects (including Points).
        # For the pyramid, use THREE.FlatShading, so the pyramid edges are not smoothed, and the camera orientations are well recognizable.
        fout.write("""

			var webGL = Detector.webgl;
			var stats, camera, controls, scene, renderer, cameras, ptcl;
			var pyrMat = new THREE.MeshStandardMaterial( { color: 0xff00ff, transparent : true, shading : THREE.FlatShading } );
			var txtMat = new THREE.MeshBasicMaterial( { color: 0xffffff, transparent : true } );

			init();
			animate();

			function init() {
				if ( ! webGL ) {
					Detector.addGetWebGLMessage( { parent : document.getElementById('info'), id : 'webGlErrorMessage' } );
					var webGlErrorMessage = document.getElementById('webGlErrorMessage');
					webGlErrorMessage.style.margin='auto';
					webGlErrorMessage.style.padding='inherit';
					webGlErrorMessage.style.width='auto';
					webGlErrorMessage.style.color='red';
					webGlErrorMessage.style.background='inherit';
				}

""")
# near- and far-plane may cause problems (objects invisible, even though in FoV. What might they help, actually? Let's not specify them.
#        fout.write("""
#				camera = new THREE.PerspectiveCamera( 60 /*vertical FoV [deg]*/, window.innerWidth / window.innerHeight /*width-to-height aspect ratio*/, .001 /*near plane*/, {farPlane} /*far plane*/ );
#				camera.position.z = {cameraPositionZ};""".format(farPlane=cameraFarPlane,cameraPositionZ=cameraPositionZ) )
        fout.write("""
				camera = new THREE.PerspectiveCamera( 60 /*vertical FoV [deg]*/, window.innerWidth / window.innerHeight /*width-to-height aspect ratio*/ );
				camera.position.z = {};\n""".format(cameraPositionZ) )
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
				var axes = new THREE.AxisHelper( axisLen );
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

        fout.write( "{}".format( len(objPts) ) )

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

dbFnOrig = r'D:\AutoPlan3D\OrientAL\12Styropor_3x4\relOri.sqlite'
if 1:
    dbFn = Path(r'D:\AutoPlan3D\OrientAL\12Styropor_3x4\relOri.sqlite').with_name('relOriCpy.sqlite')
    shutil.copyfile( dbFnOrig, dbFn )
else:
    dbFn = dbFnOrig
problemOpts = adjust.Problem.Options()
problemOpts.enable_fast_removal = True
block = adjust.Problem(problemOpts)
#loss = adjust.loss.Wrapper( adjust.loss.SoftLOne(5.) )
loss = adjust.loss.Trivial()
solveOpts = adjust.Solver.Options()
solveOpts.linear_solver_ordering = adjust.ParameterBlockOrdering()
solveOpts.linear_solver_type = adjust.LinearSolverType.SPARSE_SCHUR

pointZWkb = struct.Struct('<bIddd')
def packPointZ(pt):
    return pointZWkb.pack( 1, 1001, *pt )

with dbapi2.connect( Path(dbFn).as_uri() + '?mode=rw', uri=True ) as db:
    utils.db.initDataBase(db)
    def genObjPts():
        for row in db.execute('''
            SELECT DISTINCT objPts.id, X(pt), Y(pt), Z(pt)
            FROM objPts JOIN
                 imgObs ON imgObs.objPtId == objPts.id
            WHERE imgObs.type ISNULL OR imgObs.type == ? ''',
            [ oriental.ObservationMode.automatic ] ):
            pt = parameters.InfoTiePoint( row[0], np.array([ el for el in row[1:] ], float ) )
            yield row[0], pt
    objPts = dict( genObjPts() )
    prcs = {}
    rots = {}
    for row in db.execute('''
        SELECT id, X0, Y0, Z0, r1, r2, r3, parameterization, camId
        FROM images'''):
        prc = np.array([ el for el in row[1:4] ], float )
        prcs[row[0]] = parameters.InfoPrc( row[0], prc  )
        parameterization = adjust.EulerAngles.names[ row['parameterization'] ]
        rot = parameters.InfoEulerAngles( row[0],
                                          parametrization=parameterization,
                                          array = np.array([ el for el in row[4:7] ], float ) )
        rots[row[0]] = rot
    iors = {}
    adps = {}
    adpCols = [ str(el[1]) for el in sorted( adjust.PhotoDistortion.values.items(), key=lambda x: x[0] ) ]
    sAdpCols = [ 's_' + el for el in adpCols ]
    for row in db.execute('''
        SELECT id, x0, y0, z0, s_x0, s_y0, s_z0, normalizationRadius, reference, {}, {}
        FROM cameras
        '''.format( ', '.join(adpCols), ', '.join(sAdpCols) ) ):
        ior = parameters.InfoIor( row[0], np.array( list(row[1:4]), float ) )
        iors[row[0]] = ior
        block.AddParameterBlock( ior )
        sIor = np.array([ el for el in row[4:7] ], float )
        if ( sIor == 0. ).any():
            subset = adjust.local_parameterization.Subset( sIor.size, np.flatnonzero( sIor == 0. ).tolist() )
            block.SetParameterization( ior, subset )

        adp = adjust.parameters.InfoADP( info=row[0],
                                         normalizationRadius= row['normalizationRadius'],
                                         referencePoint     = adjust.AdpReferencePoint.names[ row['reference'] ],
                                         array              = [ row[el] for el in adpCols ] )
        adps[row[0]] = adp
        block.AddParameterBlock( adp )
        sAdp = np.array([ row[el] for el in sAdpCols ], float )
        if ( sAdp == 0. ).any():
            subset = adjust.local_parameterization.Subset( sAdp.size, np.flatnonzero( sAdp == 0. ).tolist() )
            block.SetParameterization( adp, subset )

    for id, x, y, imgId, camId, objPtId in db.execute('''
        SELECT imgObs.id, x, y, imgId, camId, objPtId
        FROM imgobs JOIN
             images ON imgobs.imgId == images.id
        WHERE type ISNULL OR type == ?
        ''', [ oriental.ObservationMode.automatic ] ):
        cost = adjust.cost.PhotoTorlegard( x, y )
        cost.data = id
        block.AddResidualBlock( cost, loss, prcs[imgId], rots[imgId], iors[camId], adps[camId], objPts[objPtId] )

    for pb in objPts.values():
        solveOpts.linear_solver_ordering.AddElementToGroup( pb, 0 )
    for cont in iors, adps, prcs, rots:
        for pb in cont.values():
            solveOpts.linear_solver_ordering.AddElementToGroup( pb, 1 )

    cam0Id = db.execute('SELECT id FROM images WHERE s_X0 == 0. AND s_Y0 == 0. AND s_Z0 == 0.').fetchone()[0]
    block.SetParameterBlockConstant( rots[cam0Id] )
    block.SetParameterBlockConstant( prcs[cam0Id] )
    cam1Id = db.execute('SELECT id FROM images WHERE s_X0 ISNULL AND s_Y0 ISNULL AND s_Z0 == 0.').fetchone()[0]
    block.SetParameterization( prcs[cam1Id], adjust.local_parameterization.UnitSphere() )
    shift = -prcs[cam0Id]
    scale = 1. / ( ( prcs[cam1Id] - prcs[cam0Id] )**2. ).sum()**.5
    for el in itertools.chain( prcs.values(), objPts.values() ):
        el += shift
        el *= scale
    summary = adjust.Solver.Summary()
    # There are many image observations, but only few control points. Control points thus contribute little to the objective function. Make sure we end-iterate, such that the sum of residuals of control points is small.
    # Appropriate weighting decreases the number of necessary iterations.
    solveOpts.function_tolerance = 1.e-16
    adjust.Solve( solveOpts, block, summary )
    if not adjust.isSuccess( summary.termination_type ) or \
        summary.num_successful_steps == 0:
        raise Exception('Adjustment failed.')

    ImgObs = namedtuple('ImgObs', 'id, x, y, ior adp rot, R t')
    imgObs = []
    for name in db.execute('''
        SELECT distinct(name)
        FROM imgObs
        WHERE type == ?''', [ oriental.ObservationMode.manual ] ):
        for id, x, y, imgId, camId, objPtId in db.execute( '''
            SELECT imgObs.id, x, y, imgId, camId, objPtId
            FROM imgObs JOIN
                 images ON imgObs.imgId == images.id
            WHERE name == ? AND
                  type == ? ''',
            [ name[0], oriental.ObservationMode.manual ] ):
            imgObs.append( ImgObs( id, x, y, iors[camId], adps[camId], rots[imgId], ori.euler2matrix( rots[imgId] ), prcs[imgId] ) )
        objPt = ori.triangulatePoints( np.array([[ imgObs[0].x, imgObs[0].y ]]),
                                       np.array([[ imgObs[1].x, imgObs[1].y ]]),
                                       imgObs[0],
                                       imgObs[1] ).ravel()
        tri = adjust.Problem()
        triLoss = adjust.loss.SoftLOne(2.)
        for imgOb in imgObs:
            tri.AddResidualBlock( adjust.cost.PhotoTorlegard( imgOb.x, imgOb.y ),
                                  triLoss,
                                  imgOb.t,
                                  imgOb.rot,
                                  imgOb.ior,
                                  imgOb.adp,
                                  objPt )
            for par in imgOb.t, imgOb.rot, imgOb.ior, imgOb.adp:
                tri.SetParameterBlockConstant( par )
        triSolveOpts = adjust.Solver.Options()
        triSolveOpts.linear_solver_type = adjust.LinearSolverType.DENSE_QR
        triSolveOpts.max_num_iterations = 100
        triSolveOpts.function_tolerance = 1.e-13
        triSolveOpts.gradient_tolerance = 1.e-13
        triSolveOpts.parameter_tolerance = 1.e-13
        adjust.Solve( triSolveOpts, tri, summary )
        if not adjust.isSuccess( summary.termination_type ) or \
            summary.num_successful_steps == 0:
            raise Exception('Forward intersection failed.')
        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.apply_loss_function = False
        residuals, = tri.Evaluate( evalOpts )
        resNormsSqr = residuals[0::2]**2 + residuals[1::2]**2

        objPtId = db.execute('''
            INSERT OR REPLACE INTO objPts( id, pt )
            VALUES( ?, GeomFromWKB(?, -1) )''',
            [ objPtId, packPointZ(objPt) ] ).lastrowid
        objPt = parameters.InfoTiePoint( objPtId, objPt )
        objPts[objPtId] = objPt
        for imgOb in imgObs:
            db.execute('''
                UPDATE imgObs
                SET objPtId = ?
                WHERE id == ?''',
                [ objPtId, imgOb.id ] )

            cost = adjust.cost.PhotoTorlegard( imgOb.x, imgOb.y )
            cost.data = imgOb.id
            block.AddResidualBlock( cost, loss, imgOb.t, imgOb.rot, imgOb.ior, imgOb.adp, objPt )
        solveOpts.linear_solver_ordering.AddElementToGroup( objPt, 0 )

    adjust.Solve( solveOpts, block, summary )
    if not adjust.isSuccess( summary.termination_type ) or \
        summary.num_successful_steps == 0:
        raise Exception('Adjustment failed.')

    db.executemany( '''
        UPDATE objpts
        SET pt = GeomFromWKB( ?, -1 ),
            s_X = NULL,
            s_Y = NULL,
            s_Z = NULL
        WHERE id == ? ''',
        ( [ packPointZ(pt), pt.info ]
          for pt in objPts.values() ) )

    def genImages():
        for prc in prcs.values():
            rot = rots[prc.info]
            yield ( *prc, *rot, prc.info )
    db.executemany( '''
        UPDATE images
        SET X0 = ?,
            Y0 = ?,
            Z0 = ?,
            r1 = ?,
            r2 = ?,
            r3 = ?,
            s_X0 = NULL,
            s_Y0 = NULL,
            s_Z0 = NULL,
            s_r1 = NULL,
            s_r2 = NULL,
            s_r3 = NULL
        WHERE id == ? ''',
        genImages() )

    def genCameras():
        for ior in iors.values():
            adp = adps[ior.info]
            iorLocPar = block.GetParameterization( ior )
            sIor = np.ones( ior.size ) * np.nan
            if iorLocPar is not None:
                sIor[iorLocPar.constancyMask] = 0.
            adpLocPar = block.GetParameterization( adp )
            sAdp = np.ones( adp.size ) * np.nan
            if adpLocPar is not None:
                sAdp[adpLocPar.constancyMask.astype(bool)] = 0.
            yield ( *ior, *sIor, *adp, *sAdp, ior.info )

    db.executemany( '''
        UPDATE cameras
        SET x0 = ?,
            y0 = ?,
            z0 = ?,
            s_x0 = ?,
            s_y0 = ?,
            s_z0 = ?,
            {},
            {}
        WHERE id == ?
        '''.format( ', '.join( '{} = ?'.format(el) for el in adpCols  ),
                    ', '.join( '{} = ?'.format(el) for el in sAdpCols ) ),
        genCameras() )

dummy=0
