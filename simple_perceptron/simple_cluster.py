# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:44:05 2017

@author: zuoquan Gong
"""

import random
from matplotlib import pyplot as plt

random.seed(0)

xnum=100
vari=0.2
x1=[random.normalvariate(1,0.2) for i in range(xnum)]
y1=[random.normalvariate(1,0.2) for i in range(xnum)]

x2=[random.normalvariate(1,0.2) for i in range(xnum)]
y2=[random.normalvariate(2,0.2) for i in range(xnum)]

x3=[random.normalvariate(2,0.2) for i in range(xnum)]
y3=[random.normalvariate(2,0.2) for i in range(xnum)]

colors=['red','green','blue']

plt.figure(figsize=(9,6))
plt.scatter(x1,y1,color='blue',marker='x',label='x1 samples')
plt.scatter(x2,y2,color='green',marker='^',label='x2 samples')
plt.scatter(x3,y3,color='red',marker='o',label='x3 samples')
plt.legend()
#plt.title("T")
plt.show()

x=x1+x2+x3
y=y1+y2+y3
plt.figure(figsize=(9,6))
plt.scatter(x,y,color='black')
plt.show()
zx=[0.2,0.3,0.4]
zy=[0.2,0.3,0.4]
plt.figure(figsize=(9,6))
plt.scatter(x,y,color='black')
plt.scatter(zx,zy,color=colors,s=250)
plt.show()

def cluster_convert(x,y,zx,zy):
    flag=0
    z_color_list=[]
    for i in range(len(x)):
        min_core=-1;
        min_distance=1000
        
        for a in range(3):
          distance=(x[i]-zx[a])**2+(y[i]-zy[a])**2
          if distance<min_distance:
              min_distance=distance
              min_core=a
        z_color_list.append(colors[min_core])
    
    '''plt.figure(figsize=(9,6))
    plt.scatter(x,y,color=z_color_list)
    plt.scatter(zx,zy,color=colors,s=250)
    plt.show()'''     
      
    for a in range(3):
        sum_x=0
        sum_y=0
        num=0
        for i in range(len(x)):
            if z_color_list[i]==colors[a]:
                num+=1
                sum_x+=x[i]
                sum_y+=y[i]
        if num==0:
            continue
        mean_x=sum_x/num;mean_y=sum_y/num
        if mean_x!=zx[a] or mean_y!=zy[a]:
            zx[a]=mean_x;zy[a]=mean_y
            flag=1
    plt.figure(figsize=(9,6))
    plt.scatter(x,y,color=z_color_list)
    plt.scatter(zx,zy,color=colors,s=250)
    plt.show()
    return flag
    
for i in range(100):
    if cluster_convert(x,y,zx,zy)==0:
        print(i,'--over')
        break