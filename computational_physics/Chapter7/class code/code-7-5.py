# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:57:20 2024

@author: lenovo
"""
import numpy as np
import pylab as pl
import random

n = 2000  # 总的分子数
B = np.ones(n) #起初所有分子在左边，其值为1
nt=10000 #经过10000次的随机选择

nr=[]  #记录随机的次数
Nl=[] #记录每次随机后左边的分子数

for i in range(1,nt):
    k=random.randint(0,1999)
    B[k]=-B[k]
    nl=0 
    for j in range(0,n): #统计左边的分子数
        if B[j]==1:
            nl=nl+1
    Nl.append(nl)
    nr.append(i)

x=[]
y=[]
for i in range(nt):  #用解析式计算右边分子数
    pr =n*0.5* (1+np.exp(-2*i/n))
    x.append(i)
    y.append(pr)

fig = pl.figure(figsize=(15,7))
ax1 =fig.add_subplot(1,2,1)
ax2 =fig.add_subplot(1,2,2)
ax1.plot(nr, Nl, 'b.')
ax2.plot(x,y, 'b-',linewidth=1.0)
ax1.set_ylabel('molecule number(simulation)', fontsize=20)
ax1.set_xlabel('random number', fontsize=20)
ax1.set_xlim(1,10000)
ax1.set_ylim(500,2000)
ax2.set_xlim(1,10000)
ax2.set_ylim(1000,2000)
ax2.set_ylabel('molecule number(analytical)', fontsize=20)
ax2.set_xlabel('random number', fontsize=20)
pl.subplots_adjust(left=0.15,bottom=0.1,top=0.9,right=0.95, \
                   hspace=0.25,wspace=0.22)
pl.show()
