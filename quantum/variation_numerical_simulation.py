#variation numerical simulation

import inspect
import tensorcircuit as tc
import random
import math
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
import time

K=tc.set_backend("tensorflow")

N=10;l=2;J=random.random();U=random.random();dt=0.0001;t=1;h=[];r=0;L_var=[];L_num=[];x_value=[];
N_num=np.zeros((int(t/dt),1));N_tro=np.zeros((int(t/dt),1));
door=[];h_door=[]
how_variation=0 #0 McLachlan 1 time-dependent
#the first 0rx,1ry,2rz,3rxx,4ryy,5rzz,6crx,7cry,8crz;the second num/ctrl+num
for k in range(l):
    for i in range(N):
        door.append([2,i])
    for i in range(N-1):
        door.append([3,i,i+1])
    for i in range(N-1):
        door.append([4,i,i+1])
    for i in range(N-1):
        door.append([5,i,i+1])
#     for i in range(N-1):
#         door.append([6,i,i+1])
#     for i in range(N-1):
#         door.append([7,i,i+1])
#     for i in range(N-1):
#         door.append([8,i,i+1])
M=len(door)

for i in range(N):
    h.append(random.random())
    h_door.append([2,i])
for i in range(N-1):
    h.append(-J);h_door.append([3,i,i+1])
    h.append(-J);h_door.append([4,i,i+1])
    h.append(U);h_door.append([5,i,i+1])
NH=len(h)

state=np.zeros(1<<N)
for i in range(1<<N):
    state[i]=random.random()
    r+=state[i]*state[i];
state=state/math.sqrt(r)
# state=np.zeros(1<<N);state[0]=1

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

def variation(ODE_theta1):
    c=tc.Circuit(N,inputs=state)
    for k in range(M):
        if door[k][0]==0:
            c.rx(door[k][1],theta=ODE_theta1[k])
        if door[k][0]==1:
            c.ry(door[k][1],theta=ODE_theta1[k])
        if door[k][0]==2:
            c.rz(door[k][1],theta=ODE_theta1[k])
        if door[k][0]==3:
            c.rxx(door[k][1],door[k][2],theta=ODE_theta1[k])
        if door[k][0]==4:
            c.ryy(door[k][1],door[k][2],theta=ODE_theta1[k])
        if door[k][0]==5:
            c.rzz(door[k][1],door[k][2],theta=ODE_theta1[k])
        if door[k][0]==6:
            c.crx(door[k][1],door[k][2],theta=ODE_theta1[k])
        if door[k][0]==7:
            c.cry(door[k][1],door[k][2],theta=ODE_theta1[k])
        if door[k][0]==8:
            c.crz(door[k][1],door[k][2],theta=ODE_theta1[k])
    return c.state()
derivative=tc.backend.jit(tc.backend.jacfwd(variation, argnums=0))

def find_dot(x,y):
    return tc.backend.tensordot(tc.backend.conj(x),y,1)

def update(ODE_theta1):
    new_state=variation(ODE_theta1)
    new_state=np.array(new_state)
    jacobian=derivative(ODE_theta1)
    ff=tc.backend.vmap(find_dot, vectorized_argnums=0)
    ff=tc.backend.vmap(ff, vectorized_argnums=1)
    jacobian=tc.backend.transpose(jacobian)
    A=tc.backend.real(ff(jacobian,jacobian))
    C=tc.backend.imag(ff(jacobian,tc.array_to_tensor(np.tile(H@new_state,(M,1)))))
    ODE_dtheta=np.linalg.pinv(A).dot(C[0])
    ODE_theta1+=ODE_dtheta*dt
    print(ODE_dtheta)
    return ODE_theta1
    
def simulation(ODE_theta1):
    c=tc.Circuit(N,inputs=state)
    for k in range(M):
        if door[k][0]==0:
            c.rx(door[k][1],theta=ODE_theta1[k])
        if door[k][0]==1:
            c.ry(door[k][1],theta=ODE_theta1[k])
        if door[k][0]==2:
            c.rz(door[k][1],theta=ODE_theta1[k])
        if door[k][0]==3:
            c.rxx(door[k][1],door[k][2],theta=ODE_theta1[k])
        if door[k][0]==4:
            c.ryy(door[k][1],door[k][2],theta=ODE_theta1[k])
        if door[k][0]==5:
            c.rzz(door[k][1],door[k][2],theta=ODE_theta1[k])
    return c

time0=time.time()
f=np.ones(M)*(-0.5j)
ODE_theta=tc.array_to_tensor(np.zeros(M))
A=np.zeros((M,M));C=np.zeros(M)
for T in range(int(t/dt)):
    ODE_theta=update(ODE_theta)
    c=simulation(ODE_theta)
    L_var.append(np.real(np.array(c.expectation([tc.gates.x(),[1]]))).tolist())
    ep=expm(-1j*H*(T+1)*dt)@state
    L_num.append(np.real(np.array(ep.conj().T@up_to_matrixx(1,0,1,1,0)@ep)).tolist())
    x_value.append((T+1)*dt)
    print([(T+1)*dt,L_num[T]-L_var[T]])
time1=time.time()
print(time1-time0)
plt.plot(x_value,L_var,color='green')
plt.plot(x_value,L_num,color='red')
plt.show()