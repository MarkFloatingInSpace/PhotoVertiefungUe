# -*- coding: cp1252 -*-

import numpy as np
from matplotlib import pyplot as plt

plt.figure(1, figsize=(6,5), dpi=150., tight_layout=True); plt.clf()
plt.axhline( y=0. )

size='small'
xShift = -.2

xn = np.linspace(0.,3./2.,1000)
plt.plot( xn, xn*(xn**2-1.), '-r', label='$3^{rd} degree$' )
x = 1./3**.5
y = x**3 - x
plt.scatter( x=[ x ], y=[ y ], c='r', marker='+' )
plt.annotate( xy=(x,y), xytext=(x+xShift,-.25), s='({:.2f},{:.2f})'.format(x,y), color='r', size=size )

plt.plot( xn, xn*(xn**4-1.), '-b', label='$5^{th} degree$' )
x = 1./5**.25
y = x**5 - x
plt.scatter( x=[ x ], y=[ y ], c='b', marker='+' )
plt.annotate( xy=(x,y), xytext=(x+xShift,-.8), s='({:.2f},{:.2f})'.format(x,y), color='b', size=size )


#plt.axvline( x=1.5, color='g' )
plt.ylim(top=1.5)

plt.legend(loc='best')
plt.title('Effect of radial distortion compensation')
plt.xlabel('$x_n$')
plt.ylabel('$dx_0(x_n,0)$')
plt.tight_layout()
