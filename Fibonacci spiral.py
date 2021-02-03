#!/usr/bin/env python
# coding: utf-8

# # 피보나치 나선(Fibonacci spiral)
# 

# ## 1. 피보나치 수열
# 
# 
# $a_n+a_{n+1}=a_{n+2} (n$은 자연수)

# In[9]:


def fibo(n):
    A=[1]
    if n <=1:
        pass
    

    else:
        a=0 
        b=1 
        for i in range(n-1): 
            a,b=b,a+b
            A += [b]
    return A

F=fibo(10)


# ## 2. 피보나치 사각형 그리기
# 

# ### (1) 시작좌표($P$를 찾는 방법 : 4가지 규칙으로 반복된다.
# 
# #### 4가지 규칙함수  ($S[k]$에서의 시작좌표 $P_k$($k$는 정수))
# 
# ![Fibonacci square](2-1.jpg)
# 
#   0. $P_0$ = (0,0)
#   1. $P_1$ 의 시작 좌표는 $P_0$ 에서 x값에 $F[0]$를 더한 값이다.
#   2. $P_2$ 의 시작 좌표는 $P_1$ 에서 x값에 $F[0]$을 빼고 y값에 $F[1]$을 더한 값이다.
#   3. $P_3$ 의 시작 좌표는 $P_2$ 에서 x값에 $F[3]$을 빼고 y값에 $F[1]$을 뺀 값이다.
#   4. $P_4$ 의 시작 좌표는 $P_3$ 에서 y값에 $F[4]$를 뺀 값이다.

# In[10]:


def start():         # 4가지 규칙함수 연산후의 시작좌표
    P=[[0,0]]        # 첫 정사각형 시작좌표
    X=0
    Y=0
    for i in range(1,len(F)):
        if i%4==1:
            X,Y = X+F[i-1],Y
            
        elif i%4==2:
            X,Y = X-F[i-2],Y+F[i-1]
            
        elif i%4==3:
            X,Y = X-F[i],Y-F[i-2]
            
        elif i%4==0:
            X,Y = X,Y-F[i]
        P += [[X,Y]]
    return P

S=start()


# ### (2) 시작 좌표에서 정사각형 그리기 : 각 꼭지점의 좌표를 구한다.
# 
# #### 이 때 한변의 길이는 피보나치 수열에 의해 결정된다. (한변의 길이: $F[k]$)
# ![Fibonacci square](2-2.jpg)
#    $R_x(k) =[x[k],x[k]+F[k],x[k]+F[k],x[k],x[k]]$
#    
#    
#    $R_y(k) =[y[k],y[k],y[k]+F[k],y[k]+F[k],y[k]]$

# In[11]:


x,y=zip(*S)    # List of S(starting point)    x=[0,1,0,-3,-3,...]    y=[0,0,1,0,-5,...]

import matplotlib.pyplot as plt

def R_x(n):
    for j in range(0,n+1):
        R_x = [x[j], x[j]+F[j], x[j]+F[j] ,x[j], x[j]]        #정사각형의 x좌표[0,1,2,3,4]
    return R_x
def R_y(n):
    for k in range(0,n+1):
        R_y = [y[k], y[k], y[k]+F[k], y[k]+F[k], y[k]]        #정사각형의 y좌표[0,1,2,3,4]
    return R_y

for l in range(0,len(F)):
    plt.plot(R_x(l),R_y(l),'k')    #FIbonacci Square

plt.axis('scaled')
plt.axis('off')

plt.show()


# ## 3. 피보나치 나선 그리기

# ### (1) 호의 중심좌표 찾기 : 사각형과 마찬가지로 4가지 규칙이 반복된다.
# 
# #### 호의 중심에 대한 4가지 규칙
# ![Fibonacci square](1.jpg)
# 
#     0.0번째 사각형은 2번 자리가 중심이다.
#     1.1번째 사각형은 3번 자리가 중심이다.
#     2.2번째 사각형은 4번 자리가 중심이다.
#     3.3번째 사각형은 1번 자리가 중심이다.
#     4.4번째 사각형은 2번 자리가 중심이다.
# 

# In[12]:


def AP():           #각 호의 중심 찾기
    P=[]
    for m in range(0,len(F)):
        if m%4==1:
            X = R_x(m)[3]
            Y = R_y(m)[3]
        elif m%4==2:
            X = R_x(m)[4]
            Y = R_y(m)[4]
        elif m%4==3:
            X = R_x(m)[1]
            Y = R_y(m)[1]
        else:
            X = R_x(m)[2]
            Y = R_y(m)[2]
        P += [[X,Y]]
    return P

xx,yy=zip(*AP())


# ### (2) 중심좌표에서 일정 각도만큼 호 그리기
# #### 4가지 규칙에 따라 호가그려진다.

# In[13]:


import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

for l in range(0,len(F)):
    plt.plot(R_x(l),R_y(l),'k')    #FIbonacci Square
    
    
for n in range(0,len(F)):
    
    r=F[n]
    
    if n%4==1:
        theta=np.linspace(-pi/2,0,100)
        
    elif n%4==2:
        theta=np.linspace(0,pi/2,100)
        
    elif n%4==3:
        theta=np.linspace(pi/2,pi,100)

    else:
        theta=np.linspace(pi,3*pi/2,100)
        
    x1=r*np.cos(theta)+xx[n]
    y1=r*np.sin(theta)+yy[n]    
    plt.plot(x1,y1,'r')
    

    
plt.title('Fibonacci Spiral')
plt.axis('scaled')
plt.axis('off')
plt.show()


# In[ ]:




