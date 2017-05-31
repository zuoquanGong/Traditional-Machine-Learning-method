# -*- coding: utf-8 -*-
"""
Created on Wed Mar  19 21:44:05 2017

@author: zuoquan Gong
"""

#==============================================================================
# 更新说明：此程序在运行后会在程序当前目录下创建一个GIF格式的图片（GMM.gif）记录整个更新过程
# 其次会生成一个名为GMM.pdf的文件，存储了样本点的初始化状态
#==============================================================================

import random
import math
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from matplotlib import animation as animation
import matplotlib.backends.backend_pdf as pdfgene
#==============================================================================
# 参数表
#==============================================================================
xnum=100 #样本点个数
vari=0.2 #样本点聚合度，值越大，聚合度越低
threshold=1e-8 #定义门限值
show_freq=5 #图像显示频率，每show_freq次迭代显示一次图像
origin=[[0.2,0.2],[0.3,0.3],[0.4,0.4]] #初始类中心点，可随机设置
#==============================================================================
# 数据点初始化
#==============================================================================
random.seed(1)

x1=[random.normalvariate(1.0,vari) for i in range(xnum)]
y1=[random.normalvariate(1.0,vari) for i in range(xnum)]

x2=[random.normalvariate(1.0,vari) for i in range(xnum)]
y2=[random.normalvariate(2.0,vari) for i in range(xnum)]

x3=[random.normalvariate(2.0,vari) for i in range(xnum)]
y3=[random.normalvariate(2.0,vari) for i in range(xnum)]
'''x1=[random.uniform(0,3) for i in range(xnum)]
y1=[random.uniform(0,3) for i in range(xnum)]

x2=[random.uniform(0,3) for i in range(xnum)]
y2=[random.uniform(0,3) for i in range(xnum)]

x3=[random.uniform(0,3) for i in range(xnum)]
y3=[random.uniform(0,3) for i in range(xnum)]'''

colors=['red','green','blue']

fig0=plt.figure(figsize=(9,6))
plt.scatter(x1,y1,color='blue',marker='x',label='x1 samples')
plt.scatter(x2,y2,color='green',marker='^',label='x2 samples')
plt.scatter(x3,y3,color='red',marker='o',label='x3 samples')
plt.grid(True)
plt.legend()
#plt.title("T")
plt.show()
with pdfgene.PdfPages('GMM.pdf') as pdf:##生成pdf格式文件
    pdf.savefig(fig0)


x=x1+x2+x3
y=y1+y2+y3
plt.figure(figsize=(9,6))
plt.scatter(x,y,color='black')
plt.show()
history=[]
z_history=[]
z_unit=[]

#==============================================================================
# GMM模型参数初始化
#==============================================================================
def add(x,y):
    return x+y

def gauss(x,y,miux,miuy,sigmax,sigmay,cov): #核心，二维高斯函数
    covmatrix=np.matrix([[sigmax,cov],[cov,sigmay]]) #构造协方差矩阵
    cov_inverse=np.linalg.pinv(covmatrix) #协方差的逆
    cov_det=np.linalg.det(cov_inverse)  #协方差的行列式
    if cov_det<0: ###
        cov_det=0
    e1=1/(2*math.pi)*np.sqrt(np.abs(cov_det))
    shift=np.matrix([[x-miux],[y-miuy]])
    er=-0.5*(np.transpose(shift)*cov_inverse*shift)
    ex=math.exp(er)
    return e1*ex
    
prob=[[0,0,0] for a in range(len(x))]
classes=[]
for k in range(3):
    unit=[]
    pi=1.0/3.0
    miux,miuy=origin[k][0],origin[k][1]
    sigmax=0.7
    sigmay=0.7
    cov=-0.6
    map(unit.append,[pi,miux,miuy,sigmax,sigmay,cov])
    classes.append(unit)
    
for index,p in enumerate(prob):
    for i in range(len(p)):
        p1=classes[i][0]*gauss(x[index],y[index],classes[i][1],classes[i][2],classes[i][3],classes[i][4],classes[i][5])
        pall=reduce(add,[classes[a][0]*gauss(x[index],y[index],classes[a][1],classes[a][2],classes[a][3],classes[a][4],classes[a][5]) for a in range(3)])
        p[i]=p1/pall
'''for p in prob:
    print p'''

z_color_list=[]
colors=['red','green','blue']
for i in range(len(x)):
    z_color_list.append(colors[prob[i].index(max(prob[i]))])
fig1=plt.figure(figsize=(9,6))
zx=[classes[k][1] for k in range(3)]
zy=[classes[k][2] for k in range(3)]
plt.scatter(zx,zy,color=['red','green','blue'],s=100)
for xx, yy in zip(zx, zy):
    plt.annotate(
        '(%.2f, %.2f)' %(xx, yy),
        xy=(xx, yy),
        xytext=(0, -10),
        textcoords='offset points',
        ha='center',
        va='top')
plt.scatter(x,y,color=z_color_list)
plt.show()

#==============================================================================
# GMM聚类迭代
#==============================================================================
def gmm_converter(prob,x,y,show_flag): #一、GMM 迭代更新过程
    #pi,miux,miuy,sigmax,sigmay,cov
    classes=[]
    for k in range(3): #更新高斯函数参数
        unit=[]
        s=reduce(add,[p[k] for p in prob])
        #print s
        pi=s/len(x)
        miux=reduce(add,[xi*prob[i][k] for i,xi in enumerate(x)])/s
        miuy=reduce(add,[yi*prob[i][k] for i,yi in enumerate(y)])/s
        sigmax=reduce(add,[prob[i][k]*((x[i]-miux)**2) for i in range(len(x))])/s
        sigmay=reduce(add,[prob[i][k]*((y[i]-miuy)**2) for i in range(len(x))])/s
        cov=reduce(add,[prob[i][k]*(x[i]-miux)*(y[i]-miuy) for i in range(len(x))])/s
        map(unit.append,[pi,miux,miuy,sigmax,sigmay,cov])
        classes.append(unit)
    '''for u in classes:
        print u'''
        
    for index,p in enumerate(prob): #二、更新各点属于各类的概率
        for i in range(len(p)):
            p1=classes[i][0]*gauss(x[index],y[index],classes[i][1],classes[i][2],classes[i][3],classes[i][4],classes[i][5])
            pall=reduce(add,[classes[a][0]*gauss(x[index],y[index],classes[a][1],classes[a][2],classes[a][3],classes[a][4],classes[a][5]) for a in range(3)])
            p[i]=p1/pall
    if show_flag==1:
        showplot(x,y,prob,classes)
    return [[classes[i][1],classes[i][2]] for i in range(3)]
        
def showplot(x,y,prob,classes): #迭代结果的图像化展示
    z_color_list=[]
    z_unit=[]
    zx=[classes[k][1] for k in range(3)]
    zy=[classes[k][2] for k in range(3)]
    z_unit.append(zx)
    z_unit.append(zy)
    z_history.append(z_unit)
    for i in range(len(x)):
        z_color_list.append(colors[prob[i].index(max(prob[i]))])
    history.append(z_color_list)
      
counter=0
variation=1.0
now=[[1.5,2.0],[2.0,1.0],[1.5,1.0]]
while variation>threshold:
    counter+=1
    show_flag=1
    '''if counter%show_freq==0:
        show_flag=1
        print counter'''
    #print prob
    last=now
    now=gmm_converter(prob,x,y,show_flag)
    variation=1/3.0*(reduce(add,[(now[i][0]-last[i][0])**2+(now[i][1]-last[i][1])**2 for i in range(3)]))

fig=plt.figure()
ax=plt.axes(xlim=(0,3), ylim=(0,3))
p=ax.scatter(x,y,color='black')
s=ax.scatter([],[])
ct=0
'''for i in range(len(z_history)):
    print 'ok',i
    print z_history[i][0]
    print z_history[i][1]'''
def init():
    plt.title('@zuoquan gong')
    plt.grid(True)
    return p,s
def update(i):
    
    '''global ct
    
    zx=z_history[ct][0]
    zy=z_history[ct][1]
    s.set_xdata(zx)
    s.set_ydata(zy)
    s.set_color(['red','green','blue'])
    s.set_markersize(100)
    ct+=1
    for xx, yy in zip(zx, zy):
        plt.annotate(
            '(%.2f, %.2f)' %(xx, yy),
            xy=(xx, yy),
            xytext=(0, -10),
            textcoords='offset points',
            ha='center',
            va='top')'''
    #上面注释的部分是实验失败的代码，即无法再GIF图片中显示中心点的更新过程及中心点的坐标值
    p.set_color(history[i])
    return p,s
ani=animation.FuncAnimation(fig,update,init_func=init,frames=len(history),interval=60,blit=True)
ani.save('GMM.gif', fps=5, writer='imagemagick',dpi=80)
plt.show()




























