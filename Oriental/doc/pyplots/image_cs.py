# -*- coding: cp1252 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
def image_cs():
    if 0:
        fig = plt.gcf()
        #axes = fig.gca( projection='3d' )
        axes = fig.add_subplot(111, projection='3d')
        axes.set_aspect("equal")
    
        X = np.arange(-0.5, 15.)
        xlen = len(X)
        Y = np.arange(0.5, -10., -1.)
        ylen = len(Y)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros( [ylen,xlen] )
        
        surf = axes.plot_wireframe(X, Y, Z, rstride=1, cstride=1,
                linewidth=0.1, antialiased=True, color='k')
    
        for idx,(axName,color) in enumerate( zip( ('x','y','z'),
                                                  ('r','g','b') ) ):
            a = Arrow3D([0,idx==0],
                        [0,idx==1],
                        [0,idx==2], mutation_scale=1, lw=1, arrowstyle="-|>", color=color)
            axes.add_artist(a)
            axes.text3D( (idx==0)*1.1,
                         (idx==1)*1.1,
                         (idx==2)*1.1,
                         axName, color=color, horizontalalignment='left', verticalalignment='bottom' )
        
        #grid = np.zeros( (150,100) )
        #plt.pcolormesh( grid, alpha = 0 )
        
        axes.autoscale_view()
        axes.set_title('digital image and camera coordinate system')
        axes.grid(False)
        #axes.set_axis_off()
    else:
        # 2D
        for idx,(axName,color) in enumerate( zip( ('x','y'),
                                                  ('r','g') ) ):
            plt.arrow(0, 0,
                      (idx==0)*5,
                      (idx==1)*5,
                      color=color,
                      width=0.1)
                       
            plt.text((idx==0)*6,
                     (idx==1)*6,
                     '+' + axName,
                     fontsize=16,
                     color=color, horizontalalignment='center', verticalalignment='center' )
        
        # another bug?
        # plt.arrow and plt.text seem to not affect the data limits!
        # work-around: plot an invisible point.
        plt.plot(0., 7., color='none')
        
        x = np.arange(-.5, 14.5)
        y = np.arange(.5, -9.5, -1.)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros( (len(y), len(x)) )
        c = plt.pcolormesh(X, Y, Z, edgecolors='k', facecolors='none')
        # just another matplotlib bug:
        # https://github.com/matplotlib/matplotlib/issues/1302
        c._is_stroked = False

        #plt.axis('equal')
        plt.axis('scaled')
        plt.axis('off')
        plt.title('Digital Image and Camera Coordinate System')
        #plt.show()

if __name__ == '__main__':
    image_cs()