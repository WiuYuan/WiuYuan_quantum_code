#前置代码

from scipy.linalg import expm
import numpy as np
A=[[1,0],[0,1]]
A=np.array(A)
expm(-1j*A)
