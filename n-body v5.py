import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#Key simulation values
G = 2.8
h = 0.01
num = 100 # number of ittertioans to run 
softening = 0

# array for bodies in future will use some sort of data format (CSV) 
# array format: mass, x, y, z, x', y', z'

start = np.array([
    [-1, 0, 0, 4, -10, 0],
    [1, 0, 0, 0, 10, 0]
], dtype=float)

masses = np.array([250, 123])

# The calcAcc function returns an array of the accelerations of each particle in the system
# Note: each row coresponds to a particle and each column coresponds to x,y,z compenent of the acceleration
def calcAcc(tvals, uvals, t, u, mass):
    #mass = masses

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
        
    return np.hstack((u[:,3:6],acc_matrix))


#Runge Kutta Integrator
def int_rk4(f, u_init, h, num, masses):
    master_array = np.zeros((num,len(u_init),len(u_init[0]))) # declare master array 
    master_array[0] = u_init
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

# center of mass function (used to center camera)
def center_mass(t, u_rk4, masses):
    total_mass = np.sum(masses)
    center_masses = []

    for i in range(len(t)): # note u_rk4 is a 3d array, (num\frame, partcs_nums, post & vel)
        center_mass = np.sum(u_rk4[i,:,0:3] * masses) / total_mass
        center_masses.append(center_mass)

    return center_masses

#Yoshida Integrator
def int_yos(f, u_init, v0, h, num, masses):
    master_array = np.zeros((num,len(u_init),len(u_init[0]))) # declare master array 
    master_array[0] = u_init
    tvals = np.zeros(num)
    
    #Coefficients from wikipedia
    cr2 = 2 ** (1/3)
    w0 = - cr2 / (2 - cr2)
    w1 = 1 / (2 - cr2)
    c1 = w1 / 2
    c2 = (w0 + w1) / 2
    
    for i in range(num - 1):
        t  = tvals[i]
        u0 = master_array[i]
        
        u1 = u0 + h * c1 * v0 
        v1 = v0 + h * w1 * f(tvals, master_array, t, u1, masses)
        u2 = u1 + h * c2 * v1
        v2 = v1 + h * w0 * f(tvals, master_array, t, u2, masses)
        u3 = u2 + h * c2 * v2
        v3 = v2 + h * w1 * f(tvals, master_array, t, u3, masses)
        u4 = u3 + h * c1 * v3
        v0 = v3
        
        tvals[i + 1] = t + h
        master_array[i + 1] = u4
        
    return (tvals, master_array)

# run simulation 
#(t, u_rk4)=int_rk4(calcAcc, start, h, num, masses)

(t, u_rk4)=int_yos(calcAcc, start[:,0:3], start[:,3:], h, num, masses)

# function that will standarized the data for animation



# will use num varible to be the number of frames
# number of particles will be length of mass list
  
num_particles = len(masses) 

# creat figure and 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# list to store lines for particles 
lines = [ax.plot([], [], [], 'o')[0] for _ in range(num_particles)]

# set limits for plot, just find the max for each axis for both postive and negtive
ax.set_xlim([np.min(u_rk4[:,:,0]), np.max(u_rk4[:,:,0])])
ax.set_ylim([np.min(u_rk4[:,:,1]), np.max(u_rk4[:,:,1])])
ax.set_zlim([np.min(u_rk4[:,:,2]), np.max(u_rk4[:,:,2])])

# ainmation update function
def update(num, lines):
    for i in range(num_particles):
        lines[i].set_data(u_rk4[:num, i, 0], u_rk4[:num, i, 1])
        lines[i].set_3d_properties(u_rk4[:num, i, 2])
    return lines
 
# create animation
ani = animation.FuncAnimation(fig, update, frames=num, fargs=(lines,), interval=100)

plt.show()