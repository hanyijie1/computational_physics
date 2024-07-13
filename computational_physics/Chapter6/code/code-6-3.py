# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:14:56 2024

@author: lenovo
"""

import scipy.special as sc
import numpy as np
import matplotlib.pyplot as plt


x=np.linspace(-10,5,1000)

fig = plt.figure(figsize=(10,8))
plt.plot(x,sc.airy(x)[0],label='Ai(x)')
plt.plot(x,sc.airy(x)[2],label='Bi(x)')
plt.legend()
plt.grid()
plt.ylabel('y', fontsize=20)
plt.xlabel('x', fontsize=20)
plt.tick_params(labelsize=15,width=1.5,length=7)
plt.tick_params(labelsize=15,which='minor',width=1,length=4)
plt.ylim(-0.5,1)
plt.show()