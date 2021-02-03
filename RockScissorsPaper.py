#!/usr/bin/env python
# coding: utf-8

# In[4]:


#가위바위보
#'가위','바위','보' 가 아닐 경우에는 질문을 반복하시오
# 연속 두번 가위바위보가 아닌 것을 내면 자동으로 패배합니다.
#컴퓨터가 낸 답과 비교하여 승자를 선택하시오.

   # 바위:1
   # 가위:2
   # 보:3
   # 3>1
   # 2>3
   # 1>2>3>1
import random

count=0
while True:
    try:
        n=str(input('가위바위보 중 하나를 내시오:'))
        if n=='바위':
            n=1
        elif n=='가위':
            n=2
        elif n=='보':
            n=3
            
        a = random.randint(1,3)
        if a==1:
            print('컴퓨터는 바위를 냈습니다.')
        elif a==2:
            print('컴퓨터는 가위를 냈습니다.')
        else:
            print('컴퓨터는 보를 냈습니다.') 



        if n==a:
            print('무승부입니다.')
        elif a>n:
            if n==1 and a==3:
                print('컴퓨터가 이겼습니다.')
            else:
                print('당신이 이겼습니다.')
        elif n>a:
            if a==1 and n==3:
                print('당신이 이겻습니다.')
            else:
                print('컴퓨터가 이겼습니다.')

    except:
        count+=1
        print('안내면 패배합니다.')
        if count==2:
            print('안냈으니까 당신이 졌습니다.')
        else:
            continue
    
    break

