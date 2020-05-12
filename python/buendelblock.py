# -*- coding: cp1252 -*-

# Import the package `numpy` under the abbreviated name `np`.
# NumPy provides class `ndarray` http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
# which we use for vectors and matrices.
import numpy as np

# Import `linalg` from package `scipy`.
# `linalg` provides optimized algorithms for linear algebra.
from scipy import linalg

# SQLite is a data base management system.
# The whole data base typically consists of a single file only.
# Since it works in-memory, no separate process or even server is needed.
# MonoScope stores observations in an .sqlite-file.
import sqlite3.dbapi2

# h5py provides Python access to HDF5-files.
# oriental.match stores feature points and their matches in this file format.
import h5py

# matplotlib provides plotting functionality in Python.
import matplotlib.pyplot as plt
from matplotlib import patheffects
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.mplot3d import Axes3D

# OpenCV
import cv2

# oriental.adjust is the (bundle) adjustment package of OrientAL.
import oriental.adjust
# oriental.adjust.cost defines observation classes.
import oriental.adjust.cost
# oriental.adjust.loss defines loss function classes.
import oriental.adjust.loss
# oriental.adjust.local_parameterization defines local parameterizations.
import oriental.adjust.local_parameterization

# Import trigonometric functions needed for the computation of residuals and derivatives.
# For the computation of residuals, function arguments will be standard real numbers and hence,
# the functions defined in the standard package `math` could as well be used.
# However, for the computation of derivatives by automatic differentiation, custom data types
# are used that only the functions defined in oriental.adjust.cost support.
from oriental.adjust.cost import sin, cos

# Extracts feature points and matches them.
import oriental.match

import oriental.ori

# Provides graph algorithms.
import oriental.graph

# Suppress logging.
import oriental.log
oriental.log.setLogFileName('')

import oriental.utils.filePaths

""""Ja servus!"""


import os
from pathlib import Path

# The file path of the data base of image measurements done in MonoScope.
# Assumes that it lies in the directory of this script, and that the default file name is used.
dataBasePath = Path(os.path.abspath(__file__)).parent / 'MonoScope.sqlite'

# Make the directory of the data base file the current working directory
# to ensure that the relative file paths to the image files are correct.
os.chdir( dataBasePath.parent )


"""returns the rotation matrix for the parameterization alpha-zeta-kappa
    angular units: `gon`
    equivalent to:

def alzeka2R(alzeka):
    typ = type(alzeka[0])
    # Trigonometric functions expect angles in radians, so convert.
    alpha, zeta, kappa = alzeka * np.pi / 200.
    R_alpha = np.array([ [ cos(alpha), -sin(alpha),     0.    ],
                         [ sin(alpha),  cos(alpha),     0.    ],
                         [     0     ,      0     ,     1.    ] ], dtype=typ )
                         
    R_zeta  = np.array([ [  cos(zeta),      0     , sin(zeta) ],
                         [      0,          1     ,     0     ],
                         [ -sin(zeta),      0     , cos(zeta) ] ], dtype=typ )
                         
    R_kappa = np.array([ [ cos(kappa), -sin(kappa),     0     ],
                         [ sin(kappa),  cos(kappa),     0     ],
                         [     0,           0,          1     ] ], dtype=typ )
    # Use numpy.ndarray's operator `@` to execute matrix multiplications (instead of element-wise ones).
    # Note: the object coordinate system gets rotated into the camera CS by the transpose of this rotation matrix.
    return R_alpha @ R_zeta @ R_kappa
"""
alzeka2R=oriental.ori.alZeKaToRotationMatrix

# Custom observation classes that inherit from oriental.adjust.cost.AutoDiff only need to define how to compute the
# residuals, but not explicitly how to compute their derivatives, because the same definition in combination with a
# special data type and automatic differentiation is used for that.
class PerspectiveCamera( oriental.adjust.cost.AutoDiff ):
    """Observation of an object point in an image of a perspective camera.
    Angular units: gon
    Parameterization of the rotation matrix: alpha-zeta-kappa i.e.:
        1. Rotation about the z-axis (by angle alpha).
        2. Rotation about the y-axis (by angle zeta).
        3. Rotation about the z-axis (by angle kappa).
    """

    def __init__( self, x_observed, y_observed, ptName=None, phoName=None ):
        """Initialize an instance of class `PerspectiveCamera` with the passed image point coordinates."""

        # Store the image coordinates as member variables of this instance of `PerspectiveCamera`.
        self.x_observed = x_observed
        self.y_observed = y_observed
        self.ptName = ptName
        self.phoName = phoName

        # Define the number of residuals that method `Evaluate` computes.
        # In the 2-dimensional image coordinate system, these are 2 residuals per image point.
        numberOfResiduals = 2

        # The number of elements of `parameterBlockSizes` defines the number of parameter blocks that method `Evaluate`
        # expects. Each element of `parameterBlockSizes` defines the number of elements that method `Evaluate` expects
        # the respective parameter block to have.
        # The number and sizes of parameter blocks defined here must match the number and sizes of parameter blocks
        # passed to block.AddResidualBlock(...).
        parameterBlockSizes = (
            3, # Projection center in the object coordinate system.
            3, # Rotation angles.
            3, # Interior orientation i.e. projection center in the camera coordinate system.
            4, # TODO Lens distortion parameters, not (yet) used here. Adapt this number to your distortion model.
            3  # Object point.
        )
        # Hence, `PerspectiveCamera` expects 5 parameter blocks, where each block consists of 3 elements
        # (i.e. unknowns), except for the block of distortion parameters.

        # Initialize the base class.
        super().__init__( numberOfResiduals, parameterBlockSizes )

    def Evaluate( self, parameterBlocks, residuals ):
        """returns the residuals of a single image point observation, based on the current values of unknown parameters.

        Reads `parameterBlocks`, and stores the computed residuals in `residuals`.

        `parameterBlocks` is a list of parameter blocks. Its length has been defined above as the length of `parameterBlockSizes`.
        Each element of `parameterBlocks` is a parameter block.
        Each parameter block is a numpy.ndarray of 1 dimension (i.e. a vector).
        Each parameter block contains as many elements as specified above by the respective element of `parameterBlockSizes`.
        Each element is of the current value of the respective unknown parameter, which generally changes with each iteration.

        `residuals` contains as many elements as specified above by `numberOfResiduals`.
        `Evaluate` is expected to set each element of `residuals`.
        """

        # Basically, `Evaluate` gets called for `numpy.ndarray`s of real numbers, to compute residuals based on the
        # current values of unknowns.
        # However, `Evaluate` also gets called for `numpy.ndarray`s whose elements are of a special data type
        # (oriental.adjust.cost.Jet), to compute the rows of the model matrix that correspond to this image observation.
        # This detail needs to be considered only when `Evaluate` itself brings real numbers into play -
        # for `PerspectiveCamera`, this is the case with the observed image coordinates.
        # For this purpose, store the data type of an arbitrary parameter.
        typ = type(parameterBlocks[0][0])

        """ Read the parameter blocks. """
        prc    = parameterBlocks[0] # Projection center.

        alzeka = parameterBlocks[1] # Rotation angles [gon], rotation matrix parameterization alpha-zeta-kappa.

        ior    = parameterBlocks[2] # Interior orientation.

        # A parameter block for distortion parameters is already defined here, but not (yet) used.
        # TODO Choose an appropriate distortion model,
        #      adapt the expected size of that parameter block in parameterBlockSizes above,
        #      and actually use that parameter block here in the computation of residuals.
        distortion = parameterBlocks[3]

        objPt = parameterBlocks[4] # Object point.


        # Elements of the interior orientation.
        x_0 = ior[0] + typ(0.) # Principal point's x-coordinate.
        y_0 = ior[1] + typ(0.) # Principal point's y-coordinate.
        c   = ior[2]           # Focal length.

        # TODO implement a distortion model and apply it.

        """ Project the object point into the image and compute the residuals. """

        # Reduce the object point coordinates to the projection center in the object coordinate system.
        objPtRed = objPt - prc

        # Compute the rotation matrix using the passed rotation angles, considering the parameterization in use.
        R = alzeka2R(alzeka)

        # Rotate the reduced object points into the camera coordinate system.
        # Mind that for this purpose, the transpose of the rotation matrix is used by convention.
        camPt = R.T.dot(objPtRed)

        # Project the reduced, rotated point into the image plane.
        x_projected = x_0 - c * camPt[0] / camPt[2]
        y_projected = y_0 - c * camPt[1] / camPt[2]

        # Compute the residuals.
        residuals[0] = self.x_observed - x_projected
        residuals[1] = self.y_observed - y_projected

        # Return True to indicate a successful computation.
        return True

def loadImageObservations():
    """returns the images points observed in MonoScope, ordered by photos.

    MonoScope creates a data base consisting of 2 tables:
    1. Table `images` lists all photos, with columns:
         `id`  : image serial number,
         `path`: image file path, relative to the data base file path.

    2. Table `imgobs` stores on each row an image point observation, with columns:
         `id`    : image observation serial number,
         `imgid` : serial number of image that corresponds to this observation,
         `name`  : point name,
         `x`     : x-coordinate in the camera coordinate system, and
         `y`     : y-coordinate in the camera coordinate system.
    The data base can be inspected and edited with standard tools,
    e.g. using $ORIENTAL_ROOT/shared/spatialite_gui.exe,
    or Python's package `sqlite3`, as done here.
    Use SQL for data base queries."""

    # Open a read-only connection to the data base that has been created with MonoScope.
    with sqlite3.dbapi2.connect(dataBasePath.as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.dbapi2.Row

        # Create a `dict` (associative array) that maps image file paths to the corresponding image point observations.
        photo_obs = dict()

        # Get the serial numbers and file paths of all images.
        photoRows = connection.execute("""
            SELECT id, path
            FROM images
        """)

        # Iterate over all result rows of the query above.
        for photoRow in photoRows:

            # Create a `dict` for the current photo that maps for each of its point observations
            # the point name (as key) to a numpy.ndarray that contains the resp. image coordinates.
            ptNames_coord = dict()

            # Get all image point observations for the current photo.
            ptRows = connection.execute("""
                SELECT name, x, y
                FROM imgobs
                WHERE imgid=?
            """, [photoRow['id']] )

            # Iterate over all result rows of the previous query.
            for ptRow in ptRows:
                ptNames_coord[ptRow['name']] = np.array([ptRow['x'],
                                                         ptRow['y']])

            # Insert the `dict` of image point observations in the current photo
            # into photo_obs, with the image file path as key.
            photo_obs[photoRow['path']] = ptNames_coord

    return photo_obs

def spatialResection(photo_obs, ior, objPts):
    """Spatial resection"""
    projectionCenters = {}
    rotationAngles = {}
    for photo, observations in photo_obs.items():
        pts = []
        for ptName, imageCoordinates in observations.items():
            objPt = objPts.get(ptName)
            if objPt is not None:
                pts.append((*( imageCoordinates * (1,-1) ), *(objPt*(1,-1,-1))))
        if len(pts) < 4:
            raise Exception(f'At least 4 corresponding image and object points are needed for an unambiguous spatial resection, but only {len(pts)} are present.')
        pts=np.array(pts)
        K=oriental.ori.cameraMatrix(ior)
        objectPoints = pts[:, 2:].reshape((-1, 1, 3)).copy()
        imagePoints = pts[:, :2].reshape((-1, 1, 2)).copy()
        rvec, tvec = cv2.solvePnP(objectPoints=objectPoints,
                                  imagePoints =imagePoints,
                                  cameraMatrix=K,
                                  distCoeffs=np.empty(0),
                                  flags=cv2.SOLVEPNP_EPNP)[1:]
        rvec, tvec = cv2.solvePnP(objectPoints=objectPoints,
                                  imagePoints =imagePoints,
                                  cameraMatrix=K,
                                  distCoeffs=np.empty(0),
                                  rvec=rvec,
                                  tvec=tvec,
                                  useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)[1:]
        res = pts[:, :2] - cv2.projectPoints(pts[:, 2:].reshape((-1, 1, 3)).copy(), rvec, tvec, K, np.empty(0))[0].squeeze()
        print('Photo {} max res norm: {:.2f}px'.format(photo, np.sum(res**2,axis=0).max()**.5))

        R, t = oriental.ori.projectionMat2oriRotTrans(np.column_stack((cv2.Rodrigues(rvec)[0], tvec)))
        projectionCenters[photo] = t
        rotationAngles[photo] = oriental.ori.rotationMatrixToAlZeKa(R.copy())

    return projectionCenters, rotationAngles

def getAutomaticTiePoints(imageFilePaths):
    """Automatically compute additional tie points.
    Tie points and their matches do not depend on exterior image orientations, and their computation is elaborate.
    Hence, re-use existing ones, if available."""
    # oriental.match.match stores its results in the following file:
    hdf5Path = dataBasePath.parent / 'features.h5'
    points = dict()
    matches = dict()
    if not hdf5Path.exists():
        print("Extract new feature points and match them. This will take some time.")
        detectionOptions = oriental.match.SiftOptions()
        detectionOptions.nAffineTilts = 0#6
        filterOptions = oriental.match.FeatureFiltOpts()
        filterOptions.nRowCol = 3, 4
        matchingOptions = oriental.match.MatchingOpts()
        oriental.match.match(imagePaths = imageFilePaths,
                             featureDetectOpts = detectionOptions,
                             featureFiltOpts = filterOptions,
                             matchingOpts = matchingOptions,
                             outDir = str(hdf5Path.parent))
    else:
        print(f"Load existing feature points and their matches from file '{hdf5Path}'")

    with h5py.File(hdf5Path, 'r') as hdf5File:
        imageName2Path = {Path(path).name : path for path in imageFilePaths}
        for imageName, hdf5Points in hdf5File['keypts'].items():
            imagePath = imageName2Path[imageName]
            points[imagePath] = np.array(hdf5Points[:, :2]).astype(float)
        for imagePairNames, hdf5Matches in hdf5File['matches'].items():
            imagePairPaths = tuple(imageName2Path[imageName] for imageName in imagePairNames.split('?'))
            matches[imagePairPaths] = np.array(hdf5Matches)

    return points, matches

def linkAutomaticTiePoints(matches):
    """Link automatic tie points that correspond to the same object point.
    This is important for object points that have been observed in more than only 2 images, because otherwise, an
    independent object point would be adjusted for each image pair, which would reduce redundancy.
    """
    imageFilePaths = {el for imagePairPaths in matches for el in imagePairPaths}
    imagePath2Index = {path : idx for idx, path in enumerate(imageFilePaths)}

    pointChains = oriental.graph.ImageFeatureTracks()
    for imagePairPaths, matchesImagePair in matches.items():
        imageIndex1 = imagePath2Index[imagePairPaths[0]]
        imageIndex2 = imagePath2Index[imagePairPaths[1]]
        for iPt1, iPt2 in matchesImagePair:
            feature1 = oriental.graph.ImageFeatureID(imageIndex1, int(iPt1))
            feature2 = oriental.graph.ImageFeatureID(imageIndex2, int(iPt2))
            pointChains.join(feature1, feature2)
    pointChains.compute()
    components = pointChains.components()

    index2ImagePath = {idx : path for path, idx in imagePath2Index.items()}
    autoImageObs = []
    for features in components.values():
        if len({feature.iImage for feature in features}) < len(features):
            continue # drop feature point chains with multiple projections into the same image
        imgPtObs = []
        for feature in features:
            imagePath = index2ImagePath[feature.iImage]
            imagePtIndex = int(feature.iFeature)
            imgPtObs.append((imagePath, imagePtIndex))
        autoImageObs.append(imgPtObs)
    return autoImageObs

def thinOutTracks( featurePoints, autoImageObs, nCols, nRows, minFeaturesPerCell ):
    print( f'Thin out feature tracks on {nCols}x{nRows} (cols x rows) grids, keeping at least {minFeaturesPerCell} features per cell with largest multiplicity' )
    imgRowsCols = 4912, 7360 # image resolution of D800
    def getRowCol( xy ):
        col = int(  xy[0] / ( imgRowsCols[1] / nCols ) )
        row = int( -xy[1] / ( imgRowsCols[0] / nRows ) )
        return row, col

    thinnedAutoImageObs = []
    featureCounts = { phoPath : np.zeros( imgRowsCols, int ) for phoPath in featurePoints }
    autoImageObs.sort( key=lambda x: len(x), reverse=True )
    for phoPathPtIds in autoImageObs:
        for phoPath, ptId in phoPathPtIds:
            xy = featurePoints[phoPath][ptId]
            row, col = getRowCol(xy)
            if featureCounts[phoPath][row,col] < minFeaturesPerCell:
                break
        else:
            continue
        for phoPath, ptId in phoPathPtIds:
            xy = featurePoints[phoPath][ptId]
            row, col = getRowCol(xy)
            featureCount = featureCounts[phoPath]
            featureCount[row,col] += 1
        thinnedAutoImageObs.append(phoPathPtIds)
    print( 'Thinned out features per image\n'
           'img\t#features\t#fullCells\n'
           '{}'.format(
           '\n'.join( '{}\t{}\t{}'.format( phoPath, counts.sum(), np.sum( counts >= minFeaturesPerCell ) )
                                           for phoPath, counts in featureCounts.items() ) ) )
    return thinnedAutoImageObs

def forwardIntersectionLinear( observations, projectionCenters, rotationAngles, ior, compResiduals=False ):
    """forward intersects image observations and returns object point coordinates that yield a minimum sum of
     squared algebraic residuals."""

    # Kinv is the inverted camera matrix:
    # Kinv @ homogeneousImageCoordinates yields image coordinates reduced to the principal point,
    # for a camera of unit focal length.
    x0, y0, c = ior
    Kinv = np.array([[-1./c,    0., x0/c],
                     [   0., -1./c, y0/c],
                     [   0.,    0.,   1.]])

    A = np.zeros(( 2*len(observations), 4 ))

    for idx,(photo,imageCoordinates) in enumerate(observations):
        imageCoordinatesNormed = Kinv @ np.r_[ imageCoordinates, 1 ]
        R = alzeka2R( rotationAngles[photo] )
        P = np.c_[ R.T, -R.T @ projectionCenters[photo] ]
        A[idx*2  ] = imageCoordinatesNormed[0]*P[2,:] - P[0,:]
        A[idx*2+1] = imageCoordinatesNormed[1]*P[2,:] - P[1,:]

    U, s, Vt = linalg.svd(A)
    X = Vt.T[:,-1]
    X = X[:3] / X[3]

    if compResiduals:
        K = np.array([[ -c,  0, x0 ],
                      [  0, -c, y0 ],
                      [  0,  0, 1. ]])
        residuals = np.zeros((len(observations),2))
        for idx,(photo,imageCoordinates) in enumerate(observations):
            R = alzeka2R( rotationAngles[photo] )
            P = np.c_[ R.T, -R.T @ projectionCenters[photo] ]
            camSys = P @ np.r_[ X, 1 ]
            if camSys[2] >= 0:
                return X, np.ones(2) * np.inf # X is behind this camera
            projection = K @ camSys
            projection = projection[:2] / projection[2]
            residuals[idx,:] = imageCoordinates - projection
        return X, residuals

    return X

def forwardIntersectionGeometric( observations, projectionCenters, rotationAngles, ior, distortion, maxLinResNorm=None ):
    """forward intersects image observations and returns object point coordinates that yield a minimum sum of
     squared geometric residuals, together with the residuals themselves.
    Returns infinite residuals in case of a rank deficit.
    """
    linRes = forwardIntersectionLinear( observations, projectionCenters, rotationAngles, ior, maxLinResNorm is not None )
    if maxLinResNorm is not None:
        X, residualsLin = linRes
        if not np.isfinite(residualsLin).all():
            return X, np.ones(2)*np.inf
        if np.sum(residualsLin ** 2, axis=1).max() > maxLinResNorm ** 2:
            return X, np.ones(2)*np.inf
    else:
        X = linRes

    loss = oriental.adjust.loss.Trivial()
    block = oriental.adjust.Problem()
    for photo, imageCoordinates in observations:
        x, y = imageCoordinates
        cost = PerspectiveCamera( x, y )
        block.AddResidualBlock( cost,
                                loss,
                                projectionCenters[photo],
                                rotationAngles[photo],
                                ior,
                                distortion,
                                X )
    evaluateOptions = oriental.adjust.Problem.EvaluateOptions()
    # Query only columns of the A-matrix that correspond to the object point coordinates.
    evaluateOptions.set_parameter_blocks( [ X ] )
    try:
        for iIter in range(10):
            l, A = block.Evaluate( evaluateOptions, residuals=True, jacobian=True )
            A = -A.toarray()
            Atl = A.T @ l
            N = A.T @ A
            C = linalg.cholesky( N, lower=False )
            delta_x = linalg.cho_solve( (C,False), Atl )
            X += delta_x
            # This is not a generally applicable stopping criterion. But at this point, it should be good enough.
            if np.abs(delta_x).max() < 1.e-3:
                break
    except linalg.LinAlgError:
        return X, np.ones(2)*np.inf

    l, = block.Evaluate( residuals=True )
    return X, l.reshape((-1,2))

def initializeObjectCoordinates( photo_obs, projectionCenters, rotationAngles, ior, distortion, objCoordinates ):
    """initialize the object point coordinates"""

    # For each point name, collect all corresponding image points.
    # For that purpose, create a `dict`
    # - with point names as keys and
    # - lists of image names and corresponding image point coordinates as values.
    ptNames_obs = dict()

    for photo, observations in photo_obs.items():
        for ptName, imageCoordinates in observations.items():
            ptNames_obs.setdefault( ptName, [] ).append( ( photo, imageCoordinates ) )

    # Determine for each non-datum point its object coordinates.
    for ptName, observations in ptNames_obs.items():

        if ptName in objCoordinates:
            # Object point is a datum point.
            continue

        if len(observations) == 0:
            raise Exception(f"Internal error: point {ptName} has not been observed in any image!")

        if len(observations) == 1:
            raise Exception(f"Point {ptName} has been observed only once (in image {observations[0][0]}), "
                             "and hence cannot be initialized")

        # Initialize the object point coordinates.
        objCoordinates[ptName], _ = forwardIntersectionGeometric( observations, projectionCenters, rotationAngles, ior, distortion )

def printParameters( projectionCenters, rotationAngles, ior, distortion, objPts ):
    def printArr(name, arr, single=False):
        print("r'{}' : np.array([{}]){}".format(name, ', '.join(f'{el:+8.3f}' for el in arr), '' if single else ','))

    printArr( 'ior', ior, single=True )

    printArr('distortion', distortion, single=True)

    print("projection centers:")
    for name, prjCtr in sorted( projectionCenters.items(), key=lambda x: x[0] ):
        printArr( name, prjCtr )

    print("rotation angles:")
    for name, rotation in sorted( rotationAngles.items(), key=lambda x: x[0] ):
        printArr( name, rotation )

    nAutoTiePoints=0
    print("object points:")
    for name, objPt in sorted(objPts.items(), key=lambda x: sortPoints(x[0])):
        if name.startswith('a'):
            nAutoTiePoints += 1
            continue
        printArr(name, objPt)
    if nAutoTiePoints:
        print(f'{nAutoTiePoints} automatic tie points not printed')

def residualStatistics( block, residuals, k=10 ):
    squaredResidualNorms = residuals[0::2]**2 + residuals[1::2]**2
    k = min(k,len(squaredResidualNorms))
    idxLargestResidualsNorms = np.argsort(squaredResidualNorms)[::-1]
    residualBlocks = block.GetResidualBlocks()
    print(f'The {k} largest residual norms:\n'
           'photo\tPoint name\tResidual norm')
    for iResidualNorm in idxLargestResidualsNorms[:k]:
        residualBlock = residualBlocks[iResidualNorm]
        cost = block.GetCostFunctionForResidualBlock(residualBlock)
        residualNorm = squaredResidualNorms[iResidualNorm]**.5
        print(f'{cost.phoName}\t{cost.ptName}\t{residualNorm:.2f}')

def residualHistogram( residuals, suffix, show ):
    tit = "Histogram of Residuals " + suffix
    fig=plt.figure(tit)
    maxAbs = np.abs( residuals ).max()
    plt.hist( residuals, bins=11, range=(-maxAbs,maxAbs) )
    plt.ylabel('Absolute Frequency')
    plt.xlabel('Residual [px]')
    plt.title(tit)
    plotDir = Path('residuals ' + suffix)
    plt.savefig(plotDir / 'residualHistogram.png')
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)

def residualPlots( block, residuals, ior, suffix, show, scale=1. ):
    plotDir = Path('residuals ' + suffix)
    print(f'Saving plots to directory "{plotDir}"')
    os.makedirs(plotDir, exist_ok=True)
    dpi = 200
    def plot(phoObsResidsNames, radius):
        plt.scatter(ior[0], ior[1], s=40, c='r', marker='+')
        plt.plot(ior[0] + radius * np.cos(np.linspace(0, 2*np.pi, 400)),
                 ior[1] + radius * np.sin(np.linspace(0, 2*np.pi, 400)), 'r')
        # Plot automatic points first, such that they cannot hide manual ones.
        sz=3
        # Plot automatic tie points in magenta, and manual ones in cyan.
        colors = np.array(['m' if ptName.startswith('a') else 'c' for (*_, ptName) in phoObsResidsNames])

        phoObsResids = np.array([el[:4] for el in phoObsResidsNames])
        plt.scatter(x=phoObsResids[:,0], y=phoObsResids[:,1], s=sz, marker='o', edgecolors='k', facecolors='k')
        lines=[]
        for color in 'm', 'c':
            act = colors == color
            if not act.any():
                continue
            lines.extend(plt.plot(np.column_stack((phoObsResids[act,0], phoObsResids[act,0] - phoObsResids[act,2] * scale)).T,
                                  np.column_stack((phoObsResids[act,1], phoObsResids[act,1] - phoObsResids[act,3] * scale)).T,
                                  color=color, linewidth=sz**.5 ))
            lines[-1].set_label('auto' if color=='m' else 'manu')
        # Representative residual norm, to be shown graphically. Cast to int for better readability.
        repResNorm = np.ceil(np.percentile(np.sum(phoObsResids[:, 2:] ** 2, axis=1), 90) ** .5).astype(int)
        ax = plt.gca()
        ax.add_artist(AnchoredSizeBar(ax.transData, size=scale * repResNorm, label=f'{repResNorm}px', loc=4, color='c'))
        plt.xlabel('x')
        plt.ylabel('y')
        return lines, ( phoObsResids[:,2:]**2 ).sum(axis=1).max() **.5

    photo_obsResidsNames = {}
    for iResidual, residualBlock in enumerate(block.GetResidualBlocks()):
        cost = block.GetCostFunctionForResidualBlock(residualBlock)
        photo_obsResidsNames.setdefault(cost.phoName, []).append(
            (cost.x_observed, cost.y_observed,
             residuals[iResidual*2], residuals[iResidual*2+1],
             cost.ptName))

    maxResidNorm = 0.
    for imagePath, obsResidsNames in photo_obsResidsNames.items():
        image = plt.imread(imagePath)
        lowResScale = float(1000 / max(image.shape[:2]))
        imgLowRes = cv2.resize(image, (0, 0), fx=lowResScale, fy=lowResScale , interpolation=cv2.INTER_AREA) # down-sample to speedup plotting
        fig=plt.figure(imagePath + ' Residuals ' + suffix, clear=True, constrained_layout=True, dpi=dpi)
        plt.imshow(imgLowRes, interpolation='nearest', cmap='gray', extent=(-.5, image.shape[1]-.5, -image.shape[0]+.5, .5))
        plt.autoscale(False)

        lines, maxResidNorm_ = plot(obsResidsNames, radius=linalg.norm(image.shape[:2])/3)
        for x, y, resX, resY, ptName in obsResidsNames:
            if not ptName.startswith('a'):
                txt = plt.text(x, y, ptName, color='k', size='small', zorder=1,
                               clip_on=True, path_effects=[patheffects.withStroke(linewidth=1, foreground='white')])
        plt.title(f"Residuals {suffix}, {scale} times enlarged. Max: {maxResidNorm_:.1f}px")
        maxResidNorm = max(maxResidNorm, maxResidNorm_)

        plt.legend(loc='best')
        plt.savefig(plotDir / ( Path(imagePath).stem + '.png' ))
        if not show:
            plt.close(fig)

    fig=plt.figure('All Residuals ' + suffix, clear=True, constrained_layout=True, dpi=dpi)
    plot([el for obsResidsNames in photo_obsResidsNames.values() for el in obsResidsNames], radius=linalg.norm(image.shape[:2])/3)
    plt.axis('image')
    plt.xlim((-.5, image.shape[1]-.5))
    plt.ylim((-image.shape[0]+.5, .5))
    plt.title(f'All Residuals {suffix}, {scale} times enlarged. Max: {maxResidNorm:.1f}px')
    plt.savefig(plotDir / 'all.png')
    print('Plotting finished')
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)

    residualHistogram(residuals, suffix, show)

def sortPoints( arg ):
    try:
        return int(arg)
    except ValueError:
        try:
            return 1.e6 + int(arg[1:])
        except ValueError:
            return 1.e7

def plot3d( titleSuffix, projectionCenters, rotationAngles, objPts ):
    def set_aspect_equal_3d(ax):
        lims=np.array(ax.get_w_lims()).reshape((3, 2))
        mid=np.mean(lims,axis=1)
        halfWidth=np.abs(lims[:,0] - mid).max()
        ax.set_xlim3d([mid[0] - halfWidth, mid[0] + halfWidth])
        ax.set_ylim3d([mid[1] - halfWidth, mid[1] + halfWidth])
        ax.set_zlim3d([mid[2] - halfWidth, mid[2] + halfWidth])

    shortNames = oriental.utils.filePaths.ShortFileNames(list(projectionCenters))
    ax = plt.figure('Object Space', clear=True).add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    ax.plot([0, 10], [0,  0], [0,  0], 'r')
    ax.plot([0,  0], [0, 10], [0,  0], 'g')
    ax.plot([0,  0], [0,  0], [0, 10], 'b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Object space ' + titleSuffix)
    c = [ 'm' if name.startswith('a') else 'c' for name in objPts ]
    ax.scatter( *zip(*objPts.values()), s=20, c=c)
    for name, objPt in objPts.items():
        if not name.startswith('a'):
            ax.text(*objPt, name)
    for name, projectionCenter in projectionCenters.items():
        ax.text(*projectionCenter, shortNames(name))
        R = alzeka2R( rotationAngles[name] )
        for idx, color in enumerate('r g b'.split()):
            endPt = [ 0, 0, 0 ]
            endPt[idx] = 2
            seg = np.array([ projectionCenter,
                             R @ endPt + projectionCenter ])
            ax.plot( seg[:,0], seg[:,1], seg[:,2], color=color )
    set_aspect_equal_3d(ax)
    plt.show(block=False)

def bundleBlock():
    photo_obs = loadImageObservations()

    # Deactivate points.
    # This may be necessary if their intersection angles are small and image orientations are still imprecise.
    # Once better estimates of interior and exterior image orientations are available, they may be activated, again.
    pointsToIgnore = [] #'28 29 30 31 32'.split()
    for obs in photo_obs.values():
        for pointName in pointsToIgnore:
            obs.pop(pointName, None)

    """ Define initial parameter values. """

    # TODO initial parameter values of the interior orientation
    # Hopefully(!) the photos have been taken shortly one after another with the same camera,
    # the same lens, and unchanged settings.
    # Hence, use the same parameters of interior orientation for all photos.
    # We need the coordinates of the projection center in the image coordinate system i.e. in [pixel].
    # For the principal point, the image center is usually a good approximation.
    # For the focal length in [pixel], we extract the focal length equivalent to 35mm film [mm]
    # from the Exif image meta data e.g. using IrfanView.
    # For the Samsung Galaxy S7's built-in camera, having taken the photos with the minimum focal length
    # and having stored full resolution images, we consider the following values:
    # - sensor / image resolution:
    #   4032 x 3024 [px] (columns x rows)
    # - nominal focal length equivalent to 35mm film:
    #   26[mm]
    # The nominal image area of 35mm film is: 36 x 24mm (width x height)
    ior = np.array([ 0.,   # x_0
                     0.,    # y_0
                     0. # c [px]
                   ], float )
    #ior = np.array([+1976.642, -1627.440, +3160.776])
    
    # TODO Distortion parameters.
    # Choose an appropriate model.
    # The number of distortion parameters stated here must match the number of distortion parameters that
    # `PerspectiveCamera` expects. The preliminary number given here only serves as a placeholder.
    distortion = np.zeros(4, float)
    #distortion = np.array([0., 0., +105.633,  -56.806,   +2.371,   +2.721])

    # `objPts` contains for each point name the object point coordinates as a vector with 3 elements.
    objPts = dict()

    # TODO Datum points.
    # Wanted datum definition:
    # - Unconstrained
    # - the +Z axis of the object coordinate system shall point to the zenith, approximately, and
    # - a coordinate plane of the object coordinate system shall be approximately
    #   parallel to the frame of the window out of which the photos were taken.
    # For an unconstrained datum definition, the 7 unknowns of a spatial similarity transform
    # can be fixed e.g. by setting 7 object point coordinates constant.
    # The distance observed with the measuring tape must be considered here.
    # Introduce object point coordinates in units of meters.
    objPts['02'] = np.array([0., 0., 0.], float)
    objPts['03'] = np.array([0., 0., 0.], float)
    objPts['01'] = np.array([0., 0., 0.], float)

    if True:
        # At the very beginning, let's derive the exterior image orientations via spatial resection.
        # For that purpose, we need object coordinates of at least 4 points manually observed in each image.
        # Since the datum points do not suffice, define additional object coordinates here.
        objPts['36'] = np.array([0., 0., 0.], float)
        projectionCenters, rotationAngles = spatialResection(photo_obs, ior, objPts)
    else:
        # Once approximate exterior image orientations have been computed, copy them below and execute this branch.

        # `projectionCenters` contains for each photo the coordinates of its projection center in the object CS.
        # photoName :  np.array([X, Y, Z])
        projectionCenters = {
            r'20200411_150033.jpg': np.array([+1.532, +0.329, -0.649]),
            r'20200411_150100.jpg': np.array([+1.146, +0.156, -0.229]),
            r'20200411_150233.jpg': np.array([+1.249, +0.271, -1.020]),
        }

        # `rotationAngles` contains for each photo the rotation angles.
        # `PerspectiveCamera` expects them in units of [gon], parameterized for alpha-zeta-kappa.
        # That is, a rotation about
        # 1. the z-axis
        # 2. the y-axis
        # 3. the z-axis.

        # photoName : np.array([alpha, zeta, kappa])
        rotationAngles = {
            r'20200411_150033.jpg': np.array([+11.822, +96.323, +98.453]),
            r'20200411_150100.jpg': np.array([+16.517, +83.882, +3.914]),
            r'20200411_150233.jpg': np.array([+22.009, +112.667, +6.396]),
        }

    # Using the initial values of interior and exterior orientations,
    # derive object point coordinates by spatial intersection.
    initializeObjectCoordinates(photo_obs, projectionCenters, rotationAngles, ior, distortion, objPts)

    print("Initial parameter values")
    printParameters(projectionCenters, rotationAngles, ior, distortion, objPts)

    """ Define the bundle block adjustment """
    # The adjustment shall minimize the squared sum of residuals (L2-norm).
    # Hence, use the 'trivial' loss-function, which simply computes the sum of squared residuals for each residual block.
    loss = oriental.adjust.loss.Trivial()

    # The central object for the definition and solution of adjustment problems:
    block = oriental.adjust.Problem()

    # Iterate over all photos.
    for photo, observations in photo_obs.items():
        # Iterate over all image observations of the current photo.
        for ptName, imageCoordinates in observations.items():
            x, y = imageCoordinates
            # Instantiate the observation class defined above with the current image coordinates, the point and photo names.
            cost = PerspectiveCamera(x, y, ptName, photo)

            # Add a residual block for the current image observation to the adjustment problem.
            block.AddResidualBlock(cost,
                                   loss,
                                   projectionCenters[photo],
                                   rotationAngles[photo],
                                   ior,
                                   distortion,
                                   objPts[ptName])

    if False:
        # TODO Once a good enough interior image orientation is known,
        # automatically compute additional tie points.
        featurePoints, featurePointMatches = getAutomaticTiePoints(list(photo_obs.keys()))
        # `featurePoints` is a `dict` that maps each image file path to an array of feature point image coordinates.
        # `featurePointMatches` is a `dict` that maps each image pair to an array of matching feature points.
        # Each row of such an array contains in the first column an index into the feature point coordinates of the
        # first photo, and likewise in the second column for the second photo.

        imageObsPerObjPt = linkAutomaticTiePoints(featurePointMatches)
        # Optionally, reduce the number of automatic tie points.
        #imageObsPerObjPt = thinOutTracks(featurePoints, imageObsPerObjPt, nCols=4, nRows=3, minFeaturesPerCell=20)
        # Add the additional image observations to the adjustment problem.
        nAutoTieObjPts = 0
        nAutoTieImgPts = 0
        for imageObs in imageObsPerObjPt:
            # Among the automatically extracted and matched feature points, outliers are to be expected,
            # which must not be introduced into the adjustment.
            # In order to detect outliers, intersect the object point and check the according image residuals.
            # NOTE: since bundle adjustment has not been executed yet, these image residuals are based on the
            # not yet adjusted image orientations!
            # Hence, their initial values must be sufficiently accurate in order to reliably detect outliers here!

            if len(imageObs) < 3:
                continue # Drop feature matches in only 2 photos, since their redundancy is low and hence, outlier residuals may be small.

            observations = []
            for photo, idxImgPt in imageObs:
                imageCoordinates = featurePoints[photo][idxImgPt]
                observations.append((photo, imageCoordinates))

            objPt, residuals = forwardIntersectionGeometric(observations, projectionCenters, rotationAngles, ior, distortion, maxLinResNorm=30)
            if not np.isfinite(residuals).all():
                # Object point triangulation failed.
                continue
            # If residuals are large, then the probability of this feature match being an outlier is high.
            # TODO But how large is *large*?
            maxResidualNorm = 1000
            residuals = np.sum(residuals**2, axis=1)**0.5
            if residuals.max() > maxResidualNorm:
                continue
            # Another way to detect outliers is e.g. to check for extreme object point coordinates.

            # Create a name for the automatically determined tie point by prefixing the index with an 'a'.
            # Hopefully, this naming schema has not been used for manual tie points already.
            ptName = f'a{nAutoTieObjPts:02}'
            assert ptName not in objPts, f"Name for automatically determined tie point already in use: {ptName}"
            nAutoTieObjPts += 1
            # Add the tie point to the list of object points.
            objPts[ptName] = objPt

            for photo, idxImgPt in imageObs:
                x, y = featurePoints[photo][idxImgPt]
                cost = PerspectiveCamera(x, y, ptName, photo)
                # For each image point, add a residual block to the adjustment problem.
                block.AddResidualBlock(cost,
                                       loss,
                                       projectionCenters[photo],
                                       rotationAngles[photo],
                                       ior,
                                       distortion,
                                       objPt)
                nAutoTieImgPts += 1

        print(f"Number of inserted automatic tie points: {nAutoTieImgPts} image points, {nAutoTieObjPts} object points")

    """ A priori residuals """
    residuals_apriori, = block.Evaluate()
    print("Sum of squared errors: {}".format(residuals_apriori @ residuals_apriori))
    residualStatistics(block, residuals_apriori, k=10)

    # Pass show=True to display the plots in windows, pass show=False to save memory.
    residualPlots(block, residuals_apriori, ior, 'a priori', show=False, scale=1.)
    plot3d('a priori', projectionCenters, rotationAngles, objPts)

    """ Datum definition """

    # Define the parameter blocks for which all elements shall be kept constant.
    # The A-matrix queried below will not contain columns for these parameter blocks.
    idsConstantBlocks = [id(el) for el in (objPts['02'],
                                           objPts['03'],
                                           ior, # TODO Set ior constant?
                                           distortion, # TODO Set distortion constant?
                                          )]

    # Since the unconstrained datum definition shall be accomplished by setting constant 7 object point coordinates,
    # the third datum point must not be set constant as a whole.
    # Instead, only set constant 1 of its coordinates to eliminate 7 unknowns in total.
    # In order to only set constant a subset of the elements of a parameter block,
    # use class `oriental.adjust.local_parameterization.Subset`.
    # Its arguments are:
    # 1. the size of the parameter block (which is 3 for object points), and
    # 2. a vector of indices of the elements to be set constant.
    # Indexing starts at 0, as usual. Hence, e.g. index 0 sets the X-coordinate constant.
    subset = oriental.adjust.local_parameterization.Subset(3, np.array([0]))
    block.SetParameterization(objPts['01'], subset)

    # Set specific distortion parameters constant?
    #subset = oriental.adjust.local_parameterization.Subset(distortion.size, np.array([0, 1]))
    #block.SetParameterization(distortion, subset)

    """ Bundle block adjustment """
    # Create a list of all parameter blocks.
    # For printing results later on, also integrate into this list the name of each block.
    # Hence, each element of this list is a pair consisting of a name and the respective parameter block.
    paramBlockNamesAndValues = (
          [("IOR", ior)]
        + [("Distortion", distortion)]
        + [(f"PRC {name}", value) for name, value in sorted(projectionCenters.items())]
        + [(f"ROT {name}", value) for name, value in sorted(rotationAngles.items())]
        + [(f"OBJ {name}", value) for name, value in sorted(objPts.items(), key=lambda x: sortPoints(x[0]))]
    )
    # Based on `paramBlockNamesAndValues` and `idsConstantBlocks`, create a list of non-constant parameter blocks.
    variableBlocks = [value for name, value in paramBlockNamesAndValues if id(value) not in idsConstantBlocks]

    evaluateOptions = oriental.adjust.Problem.EvaluateOptions()
    # Pass the parameter blocks for which the A-matrix shall contain columns, in the wanted order.
    evaluateOptions.set_parameter_blocks(variableBlocks)

    # Iteration loop. Defines a maximum number of iterations, but should be terminated before:
    for iIter in range(30):
        # l consists of the concatenated results of `PerspectiveCamera.Evaluate`
        #   i.e. observed image position minus computed projection.
        #   The order of its rows is the one in which residual blocks have been added using `AddResidualBlock`.
        # A is the matrix of partial derivatives of `l` w.r.t. the unknowns.
        #   The order of its columns is the one of `variableBlocks`.
        #   The order of its rows is the same as for `l`.
        l, A = block.Evaluate(evaluateOptions, residuals=True, jacobian=True)

        # `PerspectiveCamera` computes the partial derivatives of `l`, and not of the function value.
        # Hence, invert the signs of `A`. At the same time, convert the sparse into a dense matrix.
        A = -A.toarray()
        Atl = A.T @ l

        N = A.T @ A

        # N is symmetric and positive definite.
        # Hence, use the efficient Cholesky-factorization for solving the equation system.
        C = linalg.cholesky(N, lower=False)

        # The vector of parameter supplements.
        delta_x = linalg.cho_solve((C, False), Atl)

        # Apply the parameter supplements to the parameter blocks.
        index=0
        for paramBlock in variableBlocks:
            # Get the number of actually estimated elements for the current parameter block.
            # This is simply the parameter block's size, unless a local parameterization has been assigned to the block.
            localParameterization = block.GetParameterization(paramBlock)
            if localParameterization is None:
                # No local parameterization has been assigned to the current block.
                # Hence, supplements have been computed for all its elements.
                variableParameters = np.ones(len(paramBlock), bool)
            else:
                # A local parameterization has been assigned to this block. Hence, the A-matrix does not contain columns
                # for all of its elements, and less supplements have been computed.
                variableParameters = localParameterization.constancyMask == 0
            numberOfVariableParameters = variableParameters.sum()
            paramBlock[variableParameters] += delta_x[index : index + numberOfVariableParameters]
            index += numberOfVariableParameters

        residuals_aposteriori, = block.Evaluate()
        print("Sum of squared errors: {}".format(residuals_aposteriori @ residuals_aposteriori))

        # When shall the iteration loop be terminated?
        # TODO Define an appropriate stopping criterion here and break the loop early as soon as it is met.
    else:
        print("Warning: maximum number of iterations reached!")

    """ Results """
    # Redundancy.
    # 2 observations per image point.
    numberOfObservations = 2 * block.NumResidualBlocks()
    assert numberOfObservations == A.shape[0], 'The number of observations does not match the number of rows of the A-matrix'

    numberOfUnknowns = sum(block.ParameterBlockLocalSize(paramBlock) for paramBlock in variableBlocks)
    assert numberOfUnknowns == A.shape[1], "The number of unknowns defined by `variableBlocks` and the local parameterizations does not match the number of columns of the A-matrix"

    redundancy = numberOfObservations - numberOfUnknowns

    residuals_aposteriori, = block.Evaluate()

    print("Adjusted parameters")
    printParameters(projectionCenters, rotationAngles, ior, distortion, objPts)
    residualStatistics(block, residuals_aposteriori, k=10)
    # Pass show=True to display the plots in windows, pass show=False to save memory.
    residualPlots(block, residuals_aposteriori, ior, 'a posteriori', show=False, scale=50.)
    plot3d('a posteriori', projectionCenters, rotationAngles, objPts)

    sumOfSquaredErrors = residuals_aposteriori @ residuals_aposteriori

    sigma0 = (sumOfSquaredErrors / redundancy)**.5
    print(f"sigma0: {sigma0:.3}px")

    # Cofactor matrix of Unknowns
    # Like in the iteration loop, compute the Cholesky factorization of the N-matrix.
    # This time, however, solve for the identity matrix. Hence, the result is the inverse of N.
    A, = block.Evaluate(evaluateOptions, residuals=False, jacobian=True)
    A = -A.toarray()
    N = A.T @ A
    C = linalg.cholesky(N, lower=False)
    Qxx = linalg.cho_solve((C,False), np.eye(C.shape[0]))

    print("Standard deviations of unknowns")
    idxQxx=0
    for paramBlockName, paramBlockValue in paramBlockNamesAndValues:
        qxxDiag = np.zeros(len(paramBlockValue))
        if id(paramBlockValue) in idsConstantBlocks:
            continue
        localParameterization = block.GetParameterization(paramBlockValue)
        if localParameterization is None:
            # No local parameterization has been assigned to the current block.
            variableParameters = np.ones(len(paramBlockValue), bool)
        else:
            variableParameters = localParameterization.constancyMask == 0
        for idxElem, isVariable in enumerate(variableParameters):
            if isVariable:
                qxxDiag[idxElem] = Qxx[idxQxx, idxQxx]
                idxQxx += 1

        stdDev = sigma0 * qxxDiag**0.5
        print( "{}: {} (indices: {}:{} )".format(paramBlockName, np.array2string(stdDev, precision=3), idxQxx - variableParameters.sum(), idxQxx))

    # TODO Check the significance, correlations, etc. of the parameters of the interior orientation and lens distortion.

    # TODO Derive the wanted distances at the object and their precisions.


if __name__ == '__main__':
    bundleBlock()
    input('Hit key to exit')