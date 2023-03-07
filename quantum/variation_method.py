import inspect
import tensorcircuit as tc
import random
import math
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt

K=tc.set_backend("tensorflow")
#variation初始化，在trotter电路中假设J,U均为零简化，初态为均匀态
N=4;J=0;U=0;dt=0.2;t=2;h=np.zeros(N);r=0;L_tro=[];L_num=[];x_value=[];
N_num=np.zeros((int(t/dt),1));N_tro=np.zeros((int(t/dt),1));

for i in range(N):
    h[i]=random.random()

state=np.ones(1<<N)/(1<<(N-1))
#数值模拟
def up_to_matrixx(k,a,b,c,d):
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
for i in range(N-1):
    H+=-J*(up_to_matrixx(i,0,1,1,0)@up_to_matrixx(i+1,0,1,1,0)+up_to_matrixx(i,0,-1j,1j,0)@up_to_matrixx(i+1,0,-1j,1j,0))
    H+=U*up_to_matrixx(i,1,0,0,-1)@up_to_matrixx(i+1,1,0,0,-1)
for i in range(N):
    H+=h[i]*up_to_matrixx(i,1,0,0,-1)
#variation，M表示参数个数
M=11
def R_gate(k):#论文中普通的Rk
    if k<4:
        c.rx(k+1,theta=ODE_theta[k])
    if k>3 and k<8:
        c.rz(k-3,theta=ODE_theta[k])
    if k>7:
        c.crx(12-k,11-k,theta=ODE_theta[k])

def U_gate(k):#论文中需要控制的sigma,由R产生
    if k<4:
        c.cx(0,k+1)
    if k>3 and k<8:
        c.cz(0,k-3)
    if k>7:
        c.multicontrol(0,12-k,11-k,ctrl=[0,12-k],unitary=tc.gates._x_matrix)

def H_gate(q):#论文中需要控制的sigma,由哈密顿量产生
    c.cz(0,q+1)                                                                            #perhaps wrong
    
def find_ACkq(mod,theta_x,k,q,whi):#计算Areal,Cimag系数
    global c
    ancilla=np.array([1,np.exp(1j*theta_x)])/np.sqrt(2)
    c=tc.Circuit(N+1,inputs=np.kron(ancilla,state))
    for i in range(M):
        if i==k:
            c.x(0)
            U_gate(i)
            c.x(0)
        if whi==0 and i==q:
            U_gate(i)
        R_gate(i)
    if whi==1:
        H_gate(q)
    pstar=np.real(np.array(c.expectation([np.array([[1,1],[1,1]])/4,[0]])))
    if whi==0:#A_Real
        A[k][q]+=mod*(2*pstar-1)
    else:#C_Imag
        C[k]+=mod*(2*pstar-1)
#variation模拟，f，h与论文相同，ODE_theta表示参数，ODE_dtheta表示量子电路计算所得导数
f=np.array([-1j,-1j,-1j,-1j,-1j,-1j,-1j,-1j,-1j,-1j,-1j])/2
ODE_theta=np.zeros(M)
A=np.zeros((M,M));C=np.zeros(M);A0=np.zeros((8,8));C0=np.zeros(8)
for T in range(int(t/dt)):
    for k in range(M):
        for q in range(M):
            find_ACkq(abs(f[k]*f[q]),np.angle(f[q])-np.angle(f[k]),k,q,0)
    for k in range(M):
        for q in range(N):
            find_ACkq(abs(f[k]*h[q]),np.angle(h[q])-np.angle(f[k])-math.pi/2,k,q,1)
    ODE_dtheta=np.linalg.pinv(A).dot(C)
    print(ODE_dtheta)
    for i in range(M):
        ODE_theta[i]+=ODE_dtheta[i]*dt


