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
    [3, 5,  2,  1, -0.5, 2],
    [4, -3, 4, -2,  2.5, 5],
    [5, 3,  4, -2,    3, 1]
]

masses = np.array([1,1,1])
start = np.array(start, dtype=float) # set to float 

def calcAcc(tvals, uvals, t, u, mass):
    mass = start[:,0] # get the mass from the start array 

    # calc x,y,z postion for all particles 
    x_post = start[:,1:2]
    y_post = start[:,2:3]
    z_post = start[:,3:4]


    n_bodies = len(start)
    acc_matrix = np.zeros((n_bodies, 3))
    for i in range(n_bodies):
        acc_vec = np.zeros(3)
        for j in range(n_bodies):
            if i == j:
                continue

            i_position = start[i, 1:4]
            j_position = start[j, 1:4]
            rij = i_position - j_position

            inv_r3 = (rij[0]**2 + rij[1]**2 + rij[3]**2)**(-3/2)
            acc_vec += - rij * inv_r3 * mass[j] * G

        acc_matrix[i,:] = acc_vec

    return acc

#Runge Kutta Integrator
def int_rk4(f, u_init, h, num, masses):
    tvals = np.zeros(num)
    uvals = np.zeros((num, len(u_init)))
    uvals[0] = u_init
    
    for i in range(num - 1):
        t = tvals[i]
        u = uvals[i]
        
        k1 = f(tvals, uvals, t, u, masses)
        k2 = f(tvals, uvals, t + h/2, u + k1 * h/2, masses)
        k3 = f(tvals, uvals, t + h/2, u + k2 * h/2, masses)
        k4 = f(tvals, uvals, t + h, u + k3 * h, masses)
        
        tvals[i + 1] = t + h
        uvals[i + 1] = u + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return (tvals, uvals)

(t, u_rk4)=int_rk4(calcAcc, start, h, num, masses)
