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

start = np.array([
    [3,  5, 2,  1, -.5, 2],
    [4, -3, 4, -2, 2.5, 5],
    [5,  3, 4, -2,   3, 1]
], dtype=float)

masses = np.array([1,1,1])


def calcAcc(tvals, uvals, t, u, mass):
    mass = masses

    n_bodies = len(u)
    acc_matrix = np.zeros((n_bodies, 3))
    for i in range(n_bodies):
        acc_vec = np.zeros(3)
        for j in range(n_bodies):
            if i == j:
                continue

            i_position = u[i, 0:3]
            j_position = u[j, 0:3]
            rij = i_position - j_position

            inv_r3 = (rij[0]**2 + rij[1]**2 + rij[2]**2)**(-3/2)
            acc_vec += - rij * inv_r3 * mass[j] * G

        acc_matrix[i,:] = acc_vec

    return np.hstack((u[:][3:6],acc_matrix))

#Runge Kutta Integrator
def int_rk4(f, u_init, h, num, masses):
    master_array = np.zeros((num,len(start),len(start[0]))) # declare master array 
    tvals = np.zeros(num)

    for i in range(num - 1):
        u = master_array[i]

        t = tvals[i]
        
        k1 = f(tvals, master_array, t, u, masses)
        k2 = f(tvals, master_array, t + h/2, u + k1 * h/2, masses)
        k3 = f(tvals, master_array, t + h/2, u + k2 * h/2, masses)
        k4 = f(tvals, master_array, t + h, u + k3 * h, masses)
        
        tvals[i + 1] = t + h
        master_array[i + 1] = u + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return (tvals, master_array)

(t, u_rk4)=int_rk4(calcAcc, start, h, num, masses)
