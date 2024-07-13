# -*- coding: utf-8 -*-

"""
random walk
@author: xnS
"""


#random walk
#@author: xnS

import numpy as np
import matplotlib.pylab as pl
import random


line = input("input the maximal nubmer of walk steps:")
nstep = int(line)
print("the maximal number of walk steps is:", nstep)
###判断语句##########
if nstep > 10000:
    print('too many steps')
    
elif nstep <10:
    print('too few steps')
    
xlist = []   # create empty list 
ylist = []

x = 0.0
y = 0.0
xlist.append(x) # append the initial x value to the xlist.
ylist.append(y) # append the initial y value to the ylist.

random.seed(913) #ramdom number seed.
outfile = open('walk-traj.dat', 'w')

#####循环语句##############
for i in range(nstep):
    dx = 2*random.random()-1 # random number(-1.0,1.0)
    dy = 2*random.random()-1

    dr = np.sqrt(dx**2+dy**2) 
    dx = dx/dr #normalization
    dy = dy/dr

    x = x + dx # updating x position 
    y = y + dy  

    outfile.write('%d\t%.2f\t%.2f\n' %(i, x, y))
    
    xlist.append(x)
    ylist.append(y)
    
outfile.close()
############画图########
pl.plot(xlist[:], ylist[:],'r-o',lw=1) #visualization
pl.xlabel('x-axis')
pl.ylabel('y-axis')
pl.show()


pl.savefig('fig-1-2.png',bbox_inches='tight')


# visualization
#fig, ax0 = pl.subplots(nrows=1,ncols=3,figsize=(15,4.5))
#ax0.plot(xlist[0:], ylist[0:],color='r',ls='solid',lw=1)
#ax0.plot(xlist[0:1], ylist[0:1],'bo',label='start')
#ax0.plot(xlist[0:], ylist[0:],color='r',ls='solid',lw=1)
#ax0.plot(xlist[-2:-1], ylist[-2:-1],'go',label='end')
#ax0.set_xlim(-25,25)
#ax0.set_ylim(-25,25)
#ax0.set_xlabel(r'x-axis', fontsize=20)
#ax0.set_ylabel(r'y-axis', fontsize=20)
#ax0.legend(loc='upper left',fontsize=10)
#------------------
#pl.savefig('random-walk-traj.png',bbox_inches='tight')
#pl.show()

