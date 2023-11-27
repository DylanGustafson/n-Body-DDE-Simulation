# 2nd attempt used only yoshida and try two bodies 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Key simulation values
G = 2.8
h = 0.1
num = 300 # number of ittertioans to run 
softening = 0

# array for bodies in future will use some sort of data format (CSV) 
# array format: mass, x, y, z, x', y', z', x", y", z" (probably won't need mass or acceleration)
start = [
    [1, 3, 5, 2, 1, -0.5, 2, 0, 0, 0],
    [1, 4, -3, 4, -2, 2.5, 5, 0, 0, 0],
    [1, 5, 3, 4, -2, 3, 1, 0, 0, 0]
]

start = np.array(start, dtype=float) # set to float 

def calcAcc(start, G, softening):
    mass = start[:,0] # get the mass from the start array 

    # calc x,y,z postion for all particles 
    x_post = start[:,1:2]
    y_post = start[:,2:3]
    z_post = start[:,3:4] 

    # calc x,y,z distnace 
    x_dist = x_post.T - x_post 
    y_dist = y_post.T - y_post  
    z_dist = z_post.T - z_post 

    inv_r3 = (x_dist**2 + y_dist**2 + z_dist**2 + softening**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

    # calc x,y,z acceleration
    x_acc = G * (x_dist * inv_r3) @ mass 
    y_acc = G * (y_dist * inv_r3) @ mass
    z_acc = G * (x_dist * inv_r3) @ mass

    acc = np.hstack((x_acc, y_acc, z_acc))

    return acc

#Runge Kutta Integrator
def int_rk4(f, u_init, h, num):
    tvals = np.zeros(num)
    uvals = np.zeros((num, len(u_init)))
    uvals[0] = u_init
    
    for i in range(num - 1):
        t = tvals[i]
        u = uvals[i]
        
        k1 = f(tvals, uvals, t, u)
        k2 = f(tvals, uvals, t + h/2, u + k1 * h/2)
        k3 = f(tvals, uvals, t + h/2, u + k2 * h/2)
        k4 = f(tvals, uvals, t + h, u + k3 * h)
        
        tvals[i + 1] = t + h
        uvals[i + 1] = u + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return (tvals, uvals)

(t, u_rk4)=int_rk4(calcAcc, start, h, num)