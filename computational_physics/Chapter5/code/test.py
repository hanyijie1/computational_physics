# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:26:47 2024

@author: lenovo
"""

import pylab as pl
m=10
k=8.0/3.0
r=24

dt = 0.001
x =0.2
y =0.1
z =3
n = 60000
XT0 = []
YT0 = []
ZT0 = []
Time0 = []
t = 0.0
for i in range(n):
    x1 = m*(y-x)
    y1 = -x * z + r * x - y
    z1 = x * y - k * z
    x2 = m*(y+dt/2*y1 -x-dt/2*x1)
    y2=-(x+dt/2*x1)*(z+dt/2*z1)+r*(x+dt/2*x1)-(y+dt/2*y1)
    z2 = (x+dt/2*x1) * (y+dt/2*y1) - k * (z+dt/2*z1)
    x3 = m*(y+dt/2*y2 -x-dt/2*x2)
    y3=-(x+dt/2*x2)*(z+dt/2*z2)+r*(x+dt/2*x2)-(y+dt/2*y2)
    z3 = (x+dt/2*x2) * (y+dt/2*y2) - k * (z+dt/2*z2)
    x4 = m*(y+dt/2*y3 -x-dt/2*x3)
    y4=-(x+dt*x3)*(z+dt*z3)+r*(x+dt*x3)-(y+dt*y3)
    z4 = (x+dt*x3) * (y+dt*y3) - k * (z+dt*z3)
    x= x+dt/6*(x1+2*x2+2*x3+x4)
    y= y+dt/6*(y1+2*y2+2*y3+y4)
    z= z+dt/6*(z1+2*z2+2*z3+z4)
    t = t+dt
    XT0.append(x)
    YT0.append(y)
    ZT0.append(z)
    Time0.append(t)
fig = pl.figure(figsize=(16,4))
ax1 =fig.add_subplot(1,3,1)
ax2 =fig.add_subplot(1,3,2)
ax3 =fig.add_subplot(1,3,3)
ax1.plot(Time0, ZT0)
ax2.plot(XT0, ZT0)
ax3.plot(YT0, ZT0)
ax1.set_xlabel('t')
ax1.set_ylabel('Z')
ax1.set_xlim(0,10)
ax1.set_ylim(0,50)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax2.set_title('r=24')