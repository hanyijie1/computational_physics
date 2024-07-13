# -*- coding: utf-8 -*-
#二阶龙格库塔法#电路
import numpy as np
import pylab as pl

rc = 2.0
dt = 0.5
n = 5000
t = 0.0
q = 1.0
qt=[]
qt_analytic=[]
time = []

for i in range(n):
    t = t + dt
    q1 = q - q*dt*0.5/rc
    q = q - q1*dt/rc
    q_analytic = np.exp(-t/rc)
    qt.append(q)
    qt_analytic.append(q_analytic)
    time.append(t)

pl.plot(time,qt,'ro',label='2-order R-K')
pl.plot(time,qt_analytic,'k-',label='Analytical')
pl.xlabel('Time')
pl.ylabel('charge')
pl.xlim(0,12)
pl.ylim(-0.2,1.0)
pl.legend(loc='upper right')
pl.show()

