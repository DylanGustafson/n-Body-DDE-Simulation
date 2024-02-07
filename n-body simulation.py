import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

# Read CSV
data = pd.read_csv('C:/Users/micha/OneDrive/Documents/GitHub/n-Body-DDE-Simulation/simulation data/trial 1.csv')

G = float(data.iat[0, 0])
h = float(data.iat[0, 1]) # step size
frames = int(data.iat[0, 2]) # number of ittertioans to run
softening = float(data.iat[0, 3])

column_len = len(data['masses'])# get number of particles in system
start = data.iloc[0:column_len, 4:10].values.astype(float)
masses = data['masses'].values.astype(float)

# The calcAcc function returns an array of the accelerations of each particle in the system
# Note: each row coresponds to a particle and each column coresponds to x,y,z compenent of the acceleration
def calcAcc(time_vals, initial_postion, current_time, position_and_velocity, mass):
    #mass = masses

    n_bodies = len(position_and_velocity)
    acc_matrix = np.zeros((n_bodies, 3))
    for i in range(n_bodies):
        acc_vec = np.zeros(3)
        for j in range(n_bodies):
            if i == j:
                continue

            i_position = position_and_velocity[i, 0:3]
            j_position = position_and_velocity[j, 0:3]
            rij = i_position - j_position

            inv_r3 = (rij[0]**2 + rij[1]**2 + rij[2]**2)**(-3/2)
            acc_vec += - rij * inv_r3 * mass[j] * G

        acc_matrix[i,:] = acc_vec
        
    return np.hstack((position_and_velocity[:,3:6],acc_matrix))

#Runge Kutta Integrator
def int_rk4(f, u_init, h, frames, masses):
    master_array = np.zeros((frames,len(u_init),len(u_init[0]))) # declare master array
    master_array[0] = u_init
    time_vals = np.zeros(frames)

    for i in range(frames - 1):
        position_and_velocity = master_array[i]

        current_time = time_vals[i]

        k1 = f(time_vals, master_array, current_time, position_and_velocity, masses)
        k2 = f(time_vals, master_array, current_time + h/2, position_and_velocity + k1 * h/2, masses)
        k3 = f(time_vals, master_array, current_time + h/2, position_and_velocity + k2 * h/2, masses)
        k4 = f(time_vals, master_array, current_time + h, position_and_velocity + k3 * h, masses)

        time_vals[i + 1] = current_time + h
        master_array[i + 1] = position_and_velocity + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    return (time_vals, master_array)

#Yoshida Integrator
def int_yos(f, u_init, v0, h, frames, masses):
    master_array = np.zeros((frames,len(u_init),len(u_init[0]))) # declare master array 
    master_array[0] = u_init
    time_vals = np.zeros(frames)
    
    #Coefficients from wikipedia
    cr2 = 2 ** (1/3)
    w0 = - cr2 / (2 - cr2)
    w1 = 1 / (2 - cr2)
    c1 = w1 / 2
    c2 = (w0 + w1) / 2
    
    for i in range(frames - 1):
        current_time  = time_vals[i]
        u0 = master_array[i]
        
        u1 = u0 + h * c1 * v0 
        v1 = v0 + h * w1 * f(time_vals, master_array, current_time, u1, masses)
        u2 = u1 + h * c2 * v1
        v2 = v1 + h * w0 * f(time_vals, master_array, current_time, u2, masses)
        u3 = u2 + h * c2 * v2
        v3 = v2 + h * w1 * f(time_vals, master_array, current_time, u3, masses)
        u4 = u3 + h * c1 * v3
        v0 = v3
        
        time_vals[i + 1] = current_time + h
        master_array[i + 1] = u4
        
    return (time_vals, master_array)

# run simulation
(current_time, postion_array) = int_yos(calcAcc, start[:,0:3], start[:,3:], h, frames, masses)
#(current_time, postion_array) = int_rk4(calcAcc, start, h, frames, masses)

# Particle class 
class Particles:
    def __init__(self, postion, label, color, tail):
        self.position = postion
        self.label = label
        self.color = color if color != 'off' else 'blue'
        self.tail = tail

# empty list for particles 
Particles_list = []

# colors that look good acording to people 
colors = ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'brown', 'beige', 'maroon', 'mint', 'olive', 'coral', 'navy', ]

# number of particles
num_particles = column_len

for i in range(num_particles):
    particle_postion = postion_array[:,i,0:3]
    particle_label = f'Particle_{i+1}' # particle name 

    # if more than 20 particles turn color off
    if num_particles > 18: # if more than 18 particles turn color off 
        particle_color = 'off'
    else:
        particle_color = colors[i % len(colors)] # if more than 18 particles will wrap around

    new_particle = Particles(postion = particle_postion, label = particle_label, color = particle_color, tail = True)
    Particles_list.append(new_particle)


# creat 3D plot 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# points for each particle
points = [ax.plot([], [], [], 'o', color=particle.color)[0] for particle in Particles_list]

# tails for each particle
tails = [Line3D([], [], [], color=particle.color) for particle in Particles_list]
for tail in tails:
    ax.add_line(tail)

# animation function
def update(frames, points, tails):
    current_positions = postion_array[frames, :, 0:3]
    center_mass = np.average(current_positions, weights=masses, axis=0)

    # set graph limits around center of mass 
    max_dist = np.max(np.linalg.norm(current_positions - center_mass, axis=1))
    ax.set_xlim(center_mass[0] - max_dist, center_mass[0] + max_dist)
    ax.set_ylim(center_mass[1] - max_dist, center_mass[1] + max_dist)
    ax.set_zlim(center_mass[2] - max_dist, center_mass[2] + max_dist)

    # update points and tails 
    for i in range(num_particles):
        points[i].set_data(postion_array[frames, i, 0], postion_array[frames, i, 1])
        points[i].set_3d_properties(postion_array[frames, i, 2])
        tails[i].set_data(postion_array[:frames, i, 0], postion_array[:frames, i, 1])
        tails[i].set_3d_properties(postion_array[:frames, i, 2])

    return points + tails

# Create the animation
ani = FuncAnimation(fig, update, frames=range(frames), fargs=(points, tails), interval=100)

plt.show()