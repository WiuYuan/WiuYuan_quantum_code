import inspect
import tensorcircuit as tc
import random
import math
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt

K=tc.set_backend("tensorflow")

N=4;J=0;U=random.random();dt=0.01;t=2;h=[];r=0;L_var=[];L_num=[];x_value=[];
N_num=np.zeros((int(t/dt),1));N_tro=np.zeros((int(t/dt),1));
door=[];h_door=[]
#the first 0x,1y,2z,3cx,4cy,5cz;the second num/ctrl+num
for i in range(N):
    door.append([0,i])
    door.append([2,i])
for i in range(N):
    for j in range(i+1,N):
#         door.append([5,i,j])
        door.append([5,j,i])
M=len(door)

for i in range(N):
    h.append(random.random())
    h_door.append([2,i])
for i in range(N-1):
    h.append(-J);h_door.append([3,i,i+1])
    h.append(-J);h_door.append([4,i,i+1])
    h.append(U);h_door.append([5,i,i+1])
NH=N

state=np.zeros(1<<N)
for i in range(1<<N):
    state[i]=random.random()
    r+=state[i]*state[i];
state=state/math.sqrt(r)

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

def R_gate(k):
    if door[k][0]==0:
        c.rx(door[k][1]+1,theta=ODE_theta[k])
    if door[k][0]==1:
        c.ry(door[k][1]+1,theta=ODE_theta[k])
    if door[k][0]==2:
        c.rz(door[k][1]+1,theta=ODE_theta[k])
    if door[k][0]==3:
        c.crx(door[k][1]+1,door[k][2]+1,theta=ODE_theta[k])
    if door[k][0]==4:
        c.cry(door[k][1]+1,door[k][2]+1,theta=ODE_theta[k])
    if door[k][0]==5:
        c.crz(door[k][1]+1,door[k][2]+1,theta=ODE_theta[k])

def U_gate(k):
    if door[k][0]==0:
        c.cx(0,door[k][1]+1)
    if door[k][0]==1:
        c.cy(0,door[k][1]+1)
    if door[k][0]==2:
        c.cz(0,door[k][1]+1)
    if door[k][0]==3:
        c.multicontrol(0,door[k][1]+1,door[k][2]+1,ctrl=[0,door[k][1]+1],unitary=tc.gates._x_matrix)
    if door[k][0]==4:
        c.multicontrol(0,door[k][1]+1,door[k][2]+1,ctrl=[0,door[k][1]+1],unitary=tc.gates._y_matrix)
    if door[k][0]==5:
        c.multicontrol(0,door[k][1]+1,door[k][2]+1,ctrl=[0,door[k][1]+1],unitary=tc.gates._z_matrix)

def H_gate(q):
    if h_door[q][0]==0:
        c.cx(0,h_door[q][1]+1)
    if h_door[q][0]==1:
        c.cy(0,h_door[q][1]+1)
    if h_door[q][0]==2:
        c.cz(0,h_door[q][1]+1)
    if h_door[q][0]==3:
        c.cx(0,h_door[q][1]+1)
        c.cx(0,h_door[q][2]+1)
    if h_door[q][0]==4:
        c.cy(0,h_door[q][1]+1)
        c.cy(0,h_door[q][2]+1)
    if h_door[q][0]==5:
        c.cz(0,h_door[q][1]+1)
        c.cz(0,h_door[q][2]+1)
    
def find_ACkq(mod,theta_x,k,q,whi):
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
        
def simulation():
    global c
    c=tc.Circuit(N,inputs=state)
    for k in range(M):
        if door[k][0]==0:
            c.rx(door[k][1],theta=ODE_theta[k])
        if door[k][0]==1:
            c.ry(door[k][1],theta=ODE_theta[k])
        if door[k][0]==2:
            c.rz(door[k][1],theta=ODE_theta[k])
        if door[k][0]==3:
            c.crx(door[k][1],door[k][2],theta=ODE_theta[k])
        if door[k][0]==4:
            c.cry(door[k][1],door[k][2],theta=ODE_theta[k])
        if door[k][0]==5:
            c.crz(door[k][1],door[k][2],theta=ODE_theta[k])
    
f=np.ones(M)*(-0.5j)
ODE_theta=np.zeros(M)
A=np.zeros((M,M));C=np.zeros(M)
for T in range(int(t/dt)):
    for k in range(M):
        for q in range(M):
            find_ACkq(abs(f[k]*f[q]),np.angle(f[q])-np.angle(f[k]),k,q,0)
    for k in range(M):
        for q in range(NH):
            find_ACkq(abs(f[k]*h[q]),np.angle(h[q])-np.angle(f[k])-math.pi/2,k,q,1)
    ODE_dtheta=np.linalg.pinv(A).dot(C)
    print(ODE_dtheta)
    for i in range(M):
        ODE_theta[i]+=ODE_dtheta[i]*dt
    simulation()
    L_var.append(np.real(np.array(c.expectation([tc.gates.x(),[1]]))).tolist())
    ep=expm(-1j*H*(T+1)*dt)@state
    L_num.append(np.real(np.array(ep.conj().T@up_to_matrixx(1,0,1,1,0)@ep)).tolist())
    x_value.append((T+1)*dt)

plt.plot(x_value,L_var,color='green')
plt.plot(x_value,L_num,color='red')
plt.show()