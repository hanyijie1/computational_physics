import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# Layout
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# value
X = np.arange(-4,4,0.25)
Y = np.arange(-4,4,0.25)
X,Y = np.meshgrid(X,Y)#X-y 平面的网格
R = np.sqrt(X ** 2 + Y** 2)
Z = np.sin(R)

#plot
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# zdir :’z’/’x’/’y’示把等高线投射到哪个面
# offset : 表示等高线岛投射到指定页面的某个刻度

#
ax.contourf(X,Y,Z,zdir='z',offset=-2)
#设置z轴的显示范围，X、V轴置方式相同
ax.set_zlim(-2,2)
plt.show()