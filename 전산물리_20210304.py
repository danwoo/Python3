#!/usr/bin/env python
# coding: utf-8

# In[1]:


#파이썬 참고 사이트
#벡터,행렬연산
#연립방정식 풀이, 가우스 소거법

'''
www.python.org
www.scipy.org
www.numpy.org
'''


# In[2]:


def func(x,y):
    return x+y


# In[3]:


#lambda 함수 표기법
f= lambda x,y:x+y


# In[4]:


f(1,2)


# In[5]:


func(1,2)


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import sin,pi


# In[7]:


sin(pi/2)


# In[8]:


ex=np.array([1,0])
ey=np.array([0,1])

plt.arrow(0,0,ex[0],ex[1],head_width=0.2,color='b')
plt.arrow(0,0,ey[0],ey[1],head_width=0.2,color='r')
plt.xlim(-2,2)
plt.ylim(-2,2)


# In[10]:


#alpha 라는 매개변수를 바꾸어 가며 벡터를 사용하여 일직선을 생성합시다.
#alpha = 0 ->(0,1)
#alpha = 1 ->(1,1)
#alpha = 2 ->(2,1)

alpha=int(input('숫자 입력하시오:'))
e=np.array([alpha,1])
plt.arrow(0,0,e[0],e[1],head_width=0.2,color='g')
plt.xlim(-(alpha+1),alpha+1)
plt.ylim(-2,2)


# In[11]:


# A=[[1,alpha],
#    [0,alpha]]
# 이 연산자를 u=[1,1]에 적용했을 때 생성되는 벡터들을 그려라

u=np.array([1,1])

for alpha in np.arange(-1,1,0.2):
    A=np.array([[1,alpha],
                [0,alpha]])
    v=np.dot(A,u)
    plt.arrow(0,0,v[0],v[1],head_width=0.1,color='r')

plt.xlim(-2,2)
plt.ylim(-2,2)    
plt.show()


# In[12]:


# A*xvec=bvec
# Linear operator A(a*xvec+b*yvec)=a*A*xvec+b*A*yvec


# In[13]:


A=np.array([[1,1],
            [0,1]])
np.dot(A,ex)


# In[14]:


np.dot(A,ey)


# In[15]:


#A*x=b

A=np.array([[1,2],
            [0,3]])
b=np.array([5,4])

Ainv=np.linalg.inv(A)
x1=np.dot(Ainv,b)

x2=np.linalg.solve(A,b)

print(x1)
print(x2)


# #  $a1*30^2/2+a2*30^2/2=100$
# #  $a1*90^2/2-a2*90^2/2=100$
# 
# 
# 

# In[16]:


A=np.array([[9,9],
            [81,-81]])
b=np.array([2,2])

a=np.linalg.solve(A,b)
a


# In[17]:


t1=30
t2=90

A=np.array([[(t1**2)/2,t1**2/2],
             [(t2**2)/2,-t2**2/2]])
b=np.array([100,100])

x=np.linalg.solve(A,b)
x


# In[18]:


# A와 B가 등속운동 서로의 초기거리 100미터
#서로 반대 방향으로 걸을 때 걸리는 시간은 5초
#서로 같은 방향으로 걸을 때 걸리는 시간은 15초
#A,B속도를 찾으시오.

#5*vA+5*vB=100
#15*vA-15*vB=100

A=np.array([[5,5],
            [15,-15]])
b=np.array([100,100])

v=np.linalg.solve(A,b)
v


# In[19]:


#Aij -> Aij - lambda*Akj, j=k,k+1, ... ,n

#a21-lam*all=0
#->lam=a21/a11

#(i,j)   (행,열)
#Loop in j 열을 스캔
#Loop in i 행을 스캔    


# In[20]:


def elimination(a,b):
    n=len(a)
    for k in range(0,n-1):   #Loop in columns, k열. 첫번쨰 열부터 끝까지
        for i in range(k+1,n):   #Loop in rows, i행, 행당열+1부터 끝까지
            if a[i,k] !=0.0:     #제로가 아닐 때 연산하기
                lam=a[i,k]/a[k,k]    #Lambda 구하기
                a[i,k+1:n]=a[i,k+1:n]-lam*a[k,k+1:n]    #전체 행을 바꾸기
                b[i]=b[i]-lam*b[k]   #전체 벡터도 바꾸기
                
    return a,b

elimination(A,b)


# ## 2. Github 주소
# 
# https://github.com/danwoo/Python3

# In[35]:


#3번 문제
#ey벡터와 연산후에 종점이 y=1 선에 있는 벡터를 생성하기 위해서 
#A벡터의 (i,j)=(2,2)의 성분이 1이어야 한다.
import numpy as np

ey=np.array([0,1])

for a in np.arange(-1,1,0.2):
    A=np.array([[0,a],
                [0,1]])
    v=np.dot(A,ey)
    plt.arrow(0,0,v[0],v[1],head_width=0.1,color='r')

plt.xlim(-2,2)
plt.ylim(-2,2)    
plt.show()


# In[43]:


#4번 문제
#A를 2*2행렬로 ([a,b],[c,d])라 하면 A*u, A*v의 값을 각각 a,b,c,d로 표현하면
#a+b=3  , c+d= 2 
#a-b=-1 , c-d=-2  이므로 a,b와 c,d를 연립방정식을 푼다.
import numpy as np

ab=np.array([[1,1],
             [1,-1]])
ab0=np.array([3,-1])

cd=ab

cd0=np.array([2,-2])

a=np.linalg.solve(ab,ab0)[0]
b=np.linalg.solve(ab,ab0)[1]
c=np.linalg.solve(cd,cd0)[0]
d=np.linalg.solve(cd,cd0)[1]

A=np.array([[a,b],
            [c,d]])
A


# In[29]:


#5번 문제
#토끼+닭=40
#4*토끼+2*닭=92
import numpy as np

A=np.array([[1,1],[4,2]])
b=np.array([40,92])

x=np.linalg.solve(A,b)
print('토끼는',x[0],'마리, 닭은',x[1],'마리입니다.')

