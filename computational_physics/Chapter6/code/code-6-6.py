# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:55:27 2024

@author: lenovo
"""

import scipy.special as sc
import numpy as np
import matplotlib.pyplot as plt

I0=1
x=np.linspace(-20,20,1000)
m=1
l=2
y=sc.lpmv(m,l,x)

fig = plt.figure(figsize=(10,8))
plt.plot(x,y,label='$p^{1}_{2}(x)$')

plt.legend()
plt.grid()
plt.ylabel('p', fontsize=20)
plt.xlabel('x', fontsize=20)
plt.tick_params(labelsize=15,width=1.5,length=7)
plt.tick_params(labelsize=15,which='minor',width=1,length=4)
#plt.ylim(-0.5,1)
plt.show()