# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:11:49 2024

@author: lenovo
"""

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.gca(projection='3d')
x, y, z = axes3d.get_test_data(0.05)
ax.plot_surface(x,y, z, rstride=8, cstride=8, alpha=0.5,color='b')
cset = ax.contour(x, y, z, zdir='z', offset=-106, cmap=cm.coolwarm)
cset = ax.contour(x, y, z, zdir='x' ,offset=-40, cmap=cm.coolwarm)
cset = ax.contour(x, y, z, zdir='y', offset=40, cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_xlim(-40,40)
ax.set_ylabel('y')
ax.set_ylim(-40,40)
ax.set_zlabel('z')
ax.set_zlim(-108,188)
plt.show()