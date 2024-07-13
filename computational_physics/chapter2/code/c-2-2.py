# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:32:19 2024

@author: lenovo
"""

import turtle as t

def tree_4(length):  #送归第一种方法 以Length为基舞务件
   if length<=0:#基例，当树使长度<=8时返回
       return
   else:
       t.fd(length)
       t.left(30)
       tree_4(length-10)#树使长度每次减10
       t.right(60)
       tree_4(length-10)#树使长度每次减10
       if length-10<=0:#树使<-0时画一个都色圆点
           t.color('pink')
           t.dot(10)
           t.color('brown')
       t.left(30)
       t.bk(length)
tree_4(100)
t.hideturtle()
t.done()