# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:31:04 2024

@author: lenovo
"""

import scipy.special as sc
import numpy as np
import matplotlib.pyplot as plt

xi,eta=np.indices([200,500])/20
Ai,Aip,Bi,Bip=sc.airy(eta-(xi/2)**2)

fig = plt.figure(figsize=(10,8))

cs=plt.imshow(np.abs(Ai),cmap='jet')
cbar = fig.colorbar(cs,orientation='horizontal')

plt.ylabel('$\zeta$', fontsize=20)
plt.xlabel('$\eta$', fontsize=20)
plt.tick_params(labelsize=15,width=1.5,length=7)
plt.tick_params(labelsize=15,which='minor',width=1,length=4)

plt.show()


