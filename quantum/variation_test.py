#variation波函数比较
nnp=expm(-1j*H*t)@state
print(nnp)#数值模拟所得波函数
c=tc.Circuit(N,inputs=state)
for i in range(4):
    c.rx(i,theta=ODE_theta[i])
    c.rz(i,theta=ODE_theta[i+4])
c.crx(3,2,theta=ODE_theta[8]);c.crx(2,1,theta=ODE_theta[9]);c.crx(1,0,theta=ODE_theta[10])
nn=np.array(c.state())
print(nn)#variation所得波函数
print(nn-nnp)
