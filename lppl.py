import numpy as np
import math
import random
ln = np.log

data_npy = np.load('data.npy')
data = [np.log(data_npy[i]) for i in range(len(data_npy))]


T = []
for i in range(1000):#len(data)
    T.append(random.randint(0,len(data)-1))

def F1(t):
    return 0.063*(t**3)-5.284*(t**2)+4.887*t+412+np.random.normal(0,1)

def F2(t,A,B,C,D):
    return A*(t**B)+C*np.cos(D*t)+np.random.normal(0,1,t.shape)
#t,tc,beta,omega,phi,A= 5.37481,B= -0.0240933,C= -0.0591359
def lppl_func(t,tc,beta,omega,phi,A= 5.30542,B=-0.0150408,C=  -0.125302):
    return A + (B*np.power(tc - t, beta))*(1 + (C*np.cos((omega *np.log(tc-t))+phi)))
#    return math.e**(A + B*((tc-t)**beta)*[1+C*np.cos[omega*ln(tc-t)+phi]])
  
def gene2coef(gene):#gene由共40碼排列組合而成
    tc = np.sum(2**np.arange(4)*gene[0:4])+500#前十個gene代表A  (二進位 sum完會是0~1023 再扣-511轉成-511~+512)10個0->-5.11 10個1->+5.12 (10個bit 1024 可以代表所有組合)
    beta = (np.sum(2**np.arange(10)*gene[4:14]))/1000
    omega = (np.sum(2**np.arange(4)*gene[14:18])+2)
    phi = (np.sum(2**np.arange(12)*gene[18:30]))/1000
    return tc,beta,omega,phi




ln = np.log

T = []
for i in range(1000):#len(data)
    T.append(random.randint(0,len(data)-1))


N = 1000#可調大一點 用效率換準確率

G = 10

survive_rate = 0.01#只有百分之一的人可以活在這世界上

survive = round(N*survive_rate)#

mutation = round(N*30*0.05)#0.0001突變率->有多少基因會突變  #突變率可以調高試試看

pop = np.random.randint(0,2,(N,30))#random產生一萬個人

fit = np.zeros((N,1))



for generation in range(G):#交配10次 每次只留下1%的人再交配

    for i in range(N):

        tc,beta,omega,phi = gene2coef(pop[i,:])#第i個人的基因
        tmp = 0 
        for t in T:
            #fit[i] = np.mean(abs(F2(t,tc,beta,omega,phi)-data[t]))#跟真實的b2相減 (這個值越小越適合活在這個世界上->用fitness決定適者生存)                
            tmp += abs(lppl_func(t,tc,beta,omega,phi)-data[t])
        fit[i] = tmp/len(T)
    sortf = np.argsort(fit[:,0])#活得好的排前面，活得不好的殺掉#[(0)8,(1)4,(2)6,(3)1,(4)3]->sortf[3,4,1,2,0]
    pop = pop[sortf,:]

    #print('適者生存')

    for i in range(survive,N):#0~survive是活下來的 剩下是死掉的   #從第i個(第一個要殺掉的人)row開始，把前survive的人交配的結果取代掉i
        fid = np.random.randint(0,survive)#隨機找兩個人出來交配
        mid = np.random.randint(0,survive)
        while(fid==mid):#抽到爸媽同一個人就重抽一個媽媽
            mid = np.random.randint(0,survive)
            #print('mid',mid)
        mask = np.random.randint(0,2,(1,30))#(1*40的矩陣 裡面都是0跟1(0~2))
        son = pop[mid,:]#先把嬤嬤的基因copy給兒子 剩下再把mask是1的用爸爸的基因填
        father = pop[fid,:]
        son[mask[0,:]==1] = father[mask[0,:]==1]#son mask=1的位置跟爸爸mask=2的位置基因一樣
        pop[i,:] = son

    #基因突變

    for i in range(mutation):
        m = np.random.randint(survive,N)#random產生一個人 再random產師一個基因
        n = np.random.randint(0,30)#第M個人的第n個基因突變掉
        pop[m,n] = 1-pop[m,n]



#經過十輪交配最後活下的那一萬個人去做排序

for i in range(N):

    A,B,C,D = gene2coef(pop[i,:])
    tmp = 0 
    for t in T:
            #fit[i] = np.mean(abs(F2(t,tc,beta,omega,phi)-data[t]))#跟真實的b2相減 (這個值越小越適合活在這個世界上->用fitness決定適者生存)                
        tmp += abs(lppl_func(t,tc,beta,omega,phi)-data[t])
        fit[i] = tmp/len(T)

sortf = np.argsort(fit[:,0])

pop = pop[sortf,:]



#取出經過十輪交配最後活下的人的第0個(最後活在世上最好的人)

tc,beta,omega, phi= gene2coef(pop[0,:])

print('tc: ',tc,'beta: ',beta,'omega: ',omega,'phi: ',phi)


#%%
# =============================================================================
# tc=501
# beta=0.649
# omega=4
# phi=2.246
# 
# data_npy = np.load('data.npy')
# =============================================================================
data = [np.log(data_npy[i]) for i in range(len(data_npy))]

def lppl_function_o(t,A,B,C,tc=tc,beta=beta,omega=omega,phi=phi):#A=ln(max(data)),B=-0.619,C=0.373  np.argmax(data)
    return A + (B*np.power(tc - t, beta))*(1 + (C*np.cos((omega *np.log(tc-t))+phi)))
def lppl_function_l(t,A,B,D,tc=tc,beta=beta,omega=omega,phi=phi):#A=ln(max(data)),B=-0.619,C=0.373  np.argmax(data)
    return A + (B*np.power(tc - t, beta))+ D*np.power(tc - t, beta)*(np.cos((omega *np.log(tc-t))+phi))
AA = np.zeros((10000,3))

bb = np.zeros((10000,1))

for t in range(0,tc):#跑0~999
    bb[t] = data[t]#F1(t)
    AA[t,0] = np.power(tc - t, beta)
    AA[t,1] = np.power(tc - t, beta)*(np.cos((omega *np.log(tc-t))+phi))
    AA[t,2] = 1
   
x = np.linalg.lstsq(AA,bb)[0]#Ax=b求x (A為4次方式)(a0X^4+a1X^3+....a4)

#print(x)   


# =============================================================================
#     
# def f(x):#5.16031662*(10**-13)*(x**5)-6.41594221*(10**-10)*(x**4)+2.91031370*(10**-7)*(x**3)-5.89359944*(10**-5)*(x**2)+6.16059374*(10**-3)*(x**1)+ 4.54855528
#     #return -7.12334962*(10**-14)*(x**6)+1.15831220*(10**-10)*(x**5)-7.28283890*(10**-8)*(x**4)+2.21851468*(10**-5)*(x**3)-3.34277455*(10**-3)*(x**2)+2.24762115*(10**-1)*(x**1)+ 2.24762115*(10**-2)
#     return 5.16031662*(10**-13)*(x**5)-6.41594221*(10**-10)*(x**4)+2.91031370*(10**-7)*(x**3)-5.89359944*(10**-5)*(x**2)+6.16059374*(10**-3)*(x**1)+ 4.54855528
#     #return 2.08737921*(10**-9)*(t**3)-4.623*(10**-6)*(t**2)+2.281*(10**-3)*t+4.612569
# 
# 
# =============================================================================



import pandas as pd
import math

B = x[0][0]
D = x[1][0]
A = x[2][0]
C = D/B
print('A:',A," B:",B,' C:',C)
predict =[]
for t in range(tc):
    predict.append(math.e**(lppl_function_o( t= t,A = A,B =B, C = C)))
    #predict.append(math.e**(f(t)))#
time = np.linspace(0, len(data)-1, len(data)) 
s0 = pd.Series(np.array(time))
s1 = pd.Series(np.array(data_npy))
s2 = pd.Series(np.array(predict))

data = pd.DataFrame({'Date':s0,'Index':s1,'Fit':s2})

data = data.set_index('Date')

data.plot(figsize=(14,8))

def MAE(real, predict):
    print('MAE',np.mean(abs(real[:len(predict)]  - predict)))
final_predict =[]
for t in range(tc):
    final_predict.append(math.e**(lppl_function_o( t= t,A = A,B =B, C = C,tc=tc,beta=beta,omega=omega,phi=phi)))
MAE(data_npy,final_predict)
