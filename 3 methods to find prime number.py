#!/usr/bin/env python
# coding: utf-8

# # 소수판별법
# 
# ## 1.에라토스테네스의 체
# ### 주어진 범위에서 합성수를 지우는 방식으로 소수를 찾는다.
# 
#     1. 1은 제거
#     2. 2를 소수로 채택하고, 2의 배수를 모두 지운다.
#     3. 지워지지 않은 수 중 가장 작은 소수 3을 선택하고, 3의 배수를 모두 지운다.
#     4. 지워지지 않은 수 중 가장 작은 소수 5을 선택하고, 5의 배수를 모두 지운다.
#     5. 반복

# In[1]:


n=1000
a=[False,False]+[True]*(n-1)
primes=[]

for i in range(2,n+1):
    if a[i]:
        primes.append(i)
        for j in range(2*i,n+1,i):
            a[j]=False
            
print(primes)


# ## 2. 소수 판별법
# 
# ### 2부터 자기자신보다 1작은 수로 나눈 나머지가 0이 아닌지 판별한다.

# In[2]:


def prime(num):
    if num!=1:
        for i in range(2,num):
            if num%i==0:
                return False
    else:
        return False
    return True
    
num=int(input('Input number: '))

if prime(num):
    print('yes')
else:
    print('no')


# ## 3. n번째 소수 찾기
# 
# ### list를 생성해 list의 요소 개수가 n개가 될 때 그 숫자를 찾아낸다.

# In[3]:


def prime(n):
    primes=[2]
    a=3
    while len(primes)<=n:
        func = True
        for b in primes:
            if a%b==0:
                func=False
                break
        if func == True:
            primes.append(a)
        a=a+2
        
    return primes

print(prime(10001))

