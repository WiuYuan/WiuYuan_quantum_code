#trotter
import inspect
import tensorcircuit as tc
import random
import math
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt

K=tc.set_backend("tensorflow")
#trotter 前置初始化，包括J,U,h,初态随机化
N=6;J=random.random();U=random.random();dt=0.2;t=2;h=np.zeros((N,1));r=0;L_tro=[];L_num=[];x_value=[];
N_num=np.zeros((int(t/dt),1));N_tro=np.zeros((int(t/dt),1));

for i in range(N):
    h[i]=random.random()

state=np.zeros(1<<N)

for i in range(1<<N):
    state[i]=np.random.randint(0,2)
    r+=state[i]*state[i];
state=state/math.sqrt(r)

c=tc.Circuit(N,inputs=np.array(state))
#数值模拟
def up_to_matrixx(k,a,b,c,d):#此函数构建单个量子门到整体量子门矩阵的扩张
    I2=np.array([[1,0],[0,1]])*(1+0j);K=np.array([[a,b],[c,d]])*(1+0j);um=I2;
    if k==0:
        um=K;
    for i in range(1,N):
        if i==k:
            um=np.kron(um,K)
        else:
            um=np.kron(um,I2)
    return um
H=np.zeros((1<<N,1<<N))*1j
for i in range(N-1):#H表示哈密顿量
    H+=-J*(up_to_matrixx(i,0,1,1,0)@up_to_matrixx(i+1,0,1,1,0)+up_to_matrixx(i,0,-1j,1j,0)@up_to_matrixx(i+1,0,-1j,1j,0))
    H+=U*up_to_matrixx(i,1,0,0,-1)@up_to_matrixx(i+1,1,0,0,-1)
for i in range(N):
    H+=h[i]*up_to_matrixx(i,1,0,0,-1)
#trotter模拟
def calculate_N(theta1,theta2,theta3,num1,num2):
    #此函数实现论文中N(alpha,,beta,gama)
    c.rz(num2,theta=-math.pi/2)
    c.CNOT(num2,num1)
    c.rz(num1,theta=math.pi/2-2*theta3)
    c.ry(num2,theta=2*theta1-math.pi/2)
    c.CNOT(num1,num2)
    c.ry(num2,theta=math.pi/2-2*theta2)
    c.CNOT(num2,num1)
    c.rz(num1,theta=math.pi/2)
    return

for i in range(int(t/dt)):#实现量子电路
    for j in range(N):
        c.rz(j,theta=dt*h[j])
    for j in range(1,N-1,2):
        calculate_N(J*dt/2,J*dt/2,-U*dt/2,j,j+1)
    for j in range(0,N-1,2):
        calculate_N(J*dt,J*dt,-U*dt,j,j+1)
    for j in range(1,N-1,2):
        calculate_N(J*dt/2,J*dt/2,-U*dt/2,j,j+1)
    for j in range(N):
        c.rz(j,theta=dt*h[j])
    L_tro.append(np.real(np.array(c.expectation([tc.gates.z(),[5]]))))
    ep=expm(-1j*H*(i+1)*dt)@state
    L_num.append(np.real(ep.conj().T@up_to_matrixx(5,1,0,0,-1)@ep))
    x_value.append(J*(i+1)*dt)
    #计算两个不同的量，num和tro的后缀分别表示数值和trotter计算所得
    for j in range(int(N/2)):
        N_num[i]+=np.real(ep.conj().T@up_to_matrixx(j,1,0,0,0)@ep)
        N_tro[i]+=np.real(np.array(c.expectation([np.array([[1,0],[0,0]]),[j]])))

plt.plot(x_value,L_tro,color='green')
plt.plot(x_value,L_num,color='red')
plt.show()

plt.plot(x_value,N_tro,color='green')
plt.plot(x_value,N_num,color='red')
plt.show()
