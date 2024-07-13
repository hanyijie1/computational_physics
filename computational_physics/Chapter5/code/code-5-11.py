# -*- coding: utf-8 -*-
import numpy as np
import pylab as pl

g=9.8
xl=9.8
q = 0.5
W=2.0/3.0
dt=0.05
n = 5000
twopi = 2*np.pi

ThetaT0 = []
OmigaT0 = []
F0 = []
for F in np.arange(0.5,2.0,0.01):
    theta=0.1
    omiga=0.1
    t=0.0
    for i in range(n):
        # 迭代解微分方程
        xk1=-(g/xl)*np.sin(theta)-q*omiga+F*np.sin(W*t)
        xl1=omiga
        xk2=-(g/xl)*np.sin(theta+dt/2.*xl1)-q*(omiga+dt/2.*xk1)+F*np.sin(W*(t+dt/2))
        xl2=omiga+dt/2.*xk1
        xk3=-(g/xl)*np.sin(theta+dt/2.*xl2)-q*(omiga+dt/2.*xk2)+F*np.sin(W*(t+dt/2))
        xl3=omiga+dt/2.*xk2
        xk4=-(g/xl)*np.sin(theta+dt*xl3)-q*(omiga+dt*xk3)+F*np.sin(W*(t+dt))
        xl4=omiga+dt*xk3

        omiga=omiga+dt/6.*(xk1+2*xk2+2*xk3+xk4)
        theta=theta+dt/6.*(xl1+2*xl2+2*xl3+xl4)
        # theta值的规范
        theta=(theta+np.pi)%(2*np.pi)-np.pi
        t=t+dt
        # 在T>20之后，t取离散点N*2pi/W(固定值，取决于F，有人为确定的感觉，似乎她可以肯定周期是2pi的整数倍)。
        if abs(W*t/twopi-int(W*t/twopi)-1e-06)<0.005 and t>50:
            ThetaT0.append(theta)
            OmigaT0.append(omiga)
            F0.append(F)

fig = pl.figure(figsize=(8,5))
pl.plot(F0, ThetaT0, 'r.', label='F=0.0',ms=3.0)
pl.plot(F0, OmigaT0, 'k.', label='F=0.0',ms=3.0)
pl.ylabel(r'Theta', fontsize=20)
pl.xlabel(r'F', fontsize=20)
pl.show()
