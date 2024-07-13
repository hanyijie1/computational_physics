

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:03:47 2016

@author: wfli
"""
import numpy as np
import matplotlib.pylab as pl
#Data
nrun = 1000000
xx = np.zeros(nrun)
yy = np.zeros(nrun)
xx[0] = 0.5
yy[0] = 0.0

for irun in range(nrun-1):
    
    rand = np.random.random()
    if rand < 0.02:
       xx[irun+1] = 0.5
       yy[irun+1] = 0.27*yy[irun]
    elif rand >= 0.02 and rand < 0.17:
       xx[irun+1] = -0.139*xx[irun]+0.263*yy[irun] + 0.57
       yy[irun+1] = 0.246*xx[irun]+0.224*yy[irun] -0.036
    elif rand >= 0.17 and rand < 0.3:
        xx[irun+1] = 0.17*xx[irun]-0.215*yy[irun] + 0.408
        yy[irun+1] = 0.222*xx[irun]+0.176*yy[irun] -0.0893
    else:
        xx[irun+1] = 0.781*xx[irun]+0.034*yy[irun] + 0.1075
        yy[irun+1] = -0.032*xx[irun]+0.739*yy[irun] +0.27


pl.plot(xx, yy, 's', markersize=1)
pl.xlim(0.2,0.8)
pl.ylim(0,1)
pl.xlabel(r'X', fontsize=20)
pl.ylabel(r'Y', fontsize=20)
pl.show()
