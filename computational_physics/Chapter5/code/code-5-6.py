# -*- coding: utf-8 -*-
import pylab as pl

g=9.8
xl=9.8
dt=0.04
theta=0.1
omiga=0.
n = 4000
ThetaT0 = []
OmigaT0 = []
Time0 = []

t = 0.0
for i in range(n):
    xk1=-(g/xl)*theta
    xl1=omiga
    xk2=-(g/xl)*(theta+dt/2.*xl1)
    xl2=omiga+dt/2.*xk1
    xk3=-(g/xl)*(theta+dt/2.*xl2)
    xl3=omiga+dt/2.*xk2
    xk4=-(g/xl)*(theta+dt*xl3)
    xl4=omiga+dt*xk3

    omiga=omiga+dt/6.*(xk1+2*xk2+2*xk3+xk4)
    theta=theta+dt/6.*(xl1+2*xl2+2*xl3+xl4)
    t=t+dt
    ThetaT0.append(theta)
    OmigaT0.append(omiga)
    Time0.append(t)

fig = pl.figure(figsize=(10,4))
ax1 =fig.add_subplot(1,2,1)
ax2 =fig.add_subplot(1,2,2)
ax1.plot(Time0, ThetaT0, 'r-', label='theta0=0.1',linewidth=2.0)
ax2.plot(ThetaT0, OmigaT0, 'r-', label='theta0=0.1',linewidth=2.0)

pl.subplots_adjust(hspace=0.35,wspace=0.3)
ax1.set_ylabel('Theta', fontsize=20)
ax1.set_xlabel('Time', fontsize=20)
ax1.set_xlim(0,40)
ax1.set_ylim(-0.5,0.5)
ax2.set_xlabel(r'Theta', fontsize=20)
ax2.set_ylabel(r'Omega', fontsize=20)
ax2.set_xlim(-0.5,0.5)
ax2.set_ylim(-0.5,0.5)
pl.legend(loc='upper right')
#pl.show()

#sys.exit(0)
#---------------------------
# theta0 = 0.2
t = 0.0
g=9.8
xl=9.8
dt=0.04
theta=0.2
omiga=0.
n = 4000
ThetaT1 = []
OmigaT1 = []
Time1 = []
for i in range(n):
    xk1=-(g/xl)*theta
    xl1=omiga
    xk2=-(g/xl)*(theta+dt/2.*xl1)
    xl2=omiga+dt/2.*xk1
    xk3=-(g/xl)*(theta+dt/2.*xl2)
    xl3=omiga+dt/2.*xk2
    xk4=-(g/xl)*(theta+dt*xl3)
    xl4=omiga+dt*xk3

    omiga=omiga+dt/6.*(xk1+2*xk2+2*xk3+xk4)
    theta=theta+dt/6.*(xl1+2*xl2+2*xl3+xl4)
    t=t+dt
    ThetaT1.append(theta)
    OmigaT1.append(omiga)
    Time1.append(t)

ax1.plot(Time1, ThetaT1, 'k-', label='theta0=0.2',linewidth=2.0)
ax2.plot(ThetaT1, OmigaT1, 'k-', label='theta0=0.2',linewidth=2.0)

pl.legend(loc='upper right')
pl.show()
