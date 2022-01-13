
import numpy as np
import matplotlib.pyplot as plt

#%%
def kernel_function(u):
    """
    kernel_function: returns the function value of the Epanechnikov kernel with u as the argument
                  u: float
    """
    if abs(u) < 1:
        res = (3 / 4) *(1 - u ** 2)
    else:
        res = 0 
    return res

#%%

# simulate n = 200 datapoints. x = (x_1, ..., x_n)' and y = (y_1, ..., y_n)'
x = np.random.uniform(low = 0, high = 6, size = (200, ))      # x_i ~ Uniform(0,6)
y = np.sin(x) + np.random.normal(scale =0.5, size = (200, ))  # y_i ~ N(sin(x_i), 0.5) 

def LL(a):
    """
    LL: returns the local linear estimator which is an 2x1 array
    a:  the LL estimator is evaluated at a
    for clarification see slides 15-17 in # https://faculty.washington.edu/ezivot/econ582/nonparametricregression.pdf 
    """

    h =  1  # bandwidth
    
    # create (n x n)-matrix K
    k_i = lambda t: kernel_function((t - a) / h)                      
    vfunc = np.vectorize(k_i, otypes=[np.float])
    K = vfunc(x)
    K = np.diag(K)
    
    # create Z
    h = x - a
    Z = np.hstack((np.ones((len(x), 1)), h.reshape((len(x), 1))))
    
    # Do matrix multiplications to get our estimator = (Z'KZ)⁻¹Z'Ky
    M1 = np.linalg.inv(np.matmul(Z.transpose(), np.matmul(K,Z)))    # M1  = (Z'KZ)⁻¹  
    M2 = np.matmul(Z.transpose(), K)                                # M2  =  Z'K
    res = np.matmul(np.matmul(M1,M2),y)                             # res =  M1 * M2 * y 
    
    return res

# Create points (x2,y2) which are used to plot the LL curve 
x2 = np.random.uniform(low = 0, high = 6, size = (10000, ))
x2.sort()
y2 = LL(x2[0])
for i in range(1,len(x2)):
    y2 = np.vstack((y2,LL(x2[i])))
 
# Plot    
plt.figure() 
plt.plot(x2, y2[:,0])                # plot local linear estimator 
plt.scatter(x,y,[3]*len(y))          # plot simulated data points
plt.plot(x2, np.sin(x2), color ="red")# plot true function = sin(x)
