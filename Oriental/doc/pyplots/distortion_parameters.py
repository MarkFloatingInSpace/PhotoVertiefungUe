# -*- coding: cp1252 -*-

# bildformat: 3000x2000
# Hauptpunkt in Bildmitte
# Normalisierungsradius: 2/3 der halben Bilddiagonale

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

ext = np.array([2999.,-1999.])
x_0 = ext / 2.
norme = linalg.norm( ext ) / 3.

xn_max = ( 1.5**2 * 9 / 13 ) **.5
yn_max = xn_max * 2. / 3.

x = np.linspace( 0, ext[0], num=6*3+1 )
y = np.linspace( 0, ext[1], num=6*2+1 )
grid = x[1]-x[0]

xx,yy = np.meshgrid( x, y )
xx_n = ( xx - x_0[0] ) / norme
yy_n = ( yy - x_0[1] ) / norme
r2 = xx_n**2. + yy_n**2.

t = np.linspace( 0., 2.*np.pi, 100 )
normCircle = np.column_stack(( np.cos( t ),
                               np.sin( t ) )) * norme + x_0

plt.figure(1, figsize=(8,10), dpi=150., tight_layout=True); plt.clf()
xlims = [ float('inf'), -float('inf') ]
ylims = [ float('inf'), -float('inf') ]
names = [ 'skewness of axes',
          'scale of y-axis',
          'radial $3^{rd}$ degree',
          'radial $5^{th}$ degree',
          'tangential 1',
          'tangential 2' ]
axes = []
for iAdp in range(1,7):
    if iAdp == 1:
        dx_0 = 0. * xx_n
        dy_0 = xx_n.copy()
    elif iAdp == 2:
        dx_0 = 0. * xx_n
        dy_0 = yy_n.copy()
    elif iAdp == 3:
        dx_0 = xx_n * ( r2 - 1. )
        dy_0 = yy_n * ( r2 - 1. )
    elif iAdp == 4:
        r4 = r2**2
        dx_0 = xx_n * ( r4 - 1. )
        dy_0 = yy_n * ( r4 - 1. )
    elif iAdp == 5:
        dx_0 = r2 + 2. * xx_n**2
        dy_0 = 2. * xx_n * yy_n
    elif iAdp == 6:
        dx_0 = 2. * xx_n * yy_n
        dy_0 = r2 + 2. * yy_n**2
    else:
        raise Exception("not implemented")

    dist = ( dx_0[0,-1]**2 + \
             dy_0[0,-1]**2 )**.5
    adp = grid/dist

    dx_0 *= adp
    dy_0 *= adp
    axes.append(plt.subplot(3,2,iAdp))
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    right=False,
    left=False,
    labeltop=False,
    labelleft=False,
    labelright=False,
    labelbottom=False)

    #plt.figure(iAdp, tight_layout=True); plt.clf()
    plt.plot( xx  , yy  , '-k' )
    plt.plot( xx.T, yy.T, '-k' )
    plt.plot( xx   + dx_0  , yy   + dy_0  , '-r', linewidth=2. )
    plt.plot( xx.T + dx_0.T, yy.T + dy_0.T, '-r', linewidth=2. )

    plt.plot( normCircle[:,0], normCircle[:,1], 'b:' )
    plt.plot( x_0[0], x_0[1], 'b*' )
    plt.axis('image')

    #plt.xlabel('x')
    #plt.ylabel('y')

    plt.title( names[iAdp-1] )

    xlims = [ min(xlims[0], plt.xlim()[0]), max( xlims[1], plt.xlim()[1] ) ]
    ylims = [ min(ylims[0], plt.ylim()[0]), max( ylims[1], plt.ylim()[1] ) ]

xlims = [ xlims[0]-grid/2.,
          xlims[1]+grid/2.]
ylims = [ ylims[0]-grid/2.,
          ylims[1]+grid/2.]
for axis in axes:
    plt.sca(axis)
    plt.xlim( xlims )
    plt.ylim( ylims )

plt.tight_layout()


##plt.figure()
##plt.quiver( xx, yy, dx_0, dy_0 )
##plt.plot( normCircle[:,0], normCircle[:,1], 'b:' )
##plt.axis('image')
##plt.xlabel('x')
##plt.ylabel('y')

