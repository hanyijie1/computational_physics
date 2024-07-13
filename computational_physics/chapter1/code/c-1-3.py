# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:28:16 2023

@author: lenovo
"""

import turtle as tur

tur.setup(650,350,200,200)
tur.penup()
tur.fd(-250)
tur.pendown()
tur.pensize(25)
tur.pencolor("blue")
tur.seth(-40)

for i in range(4):
    tur.circle(40,80)
    tur.circle(-40,80)
    

tur.done()