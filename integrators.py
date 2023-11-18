import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Key simulation values
G = 2.8
h = 0.1
num = 300
start = np.zeros(4)
start[:2] = [ 2,3]      #Starting xy Position
start[2:] = [-0.5,0]    #Starting xy Velocity

#Newtonian: only uses current u values (for now)
def f_grav(tvals, uvals, t, u):
    r = u[:2]   #position vector in first half of u
    v = u[2:]   #velocity vector in second half
    
    r_prime = v
    v_prime = -r * G * (r[0]**2 + r[1]**2) ** (-3/2)
    
    u_prime = np.hstack((r_prime, v_prime))
    return u_prime

#Forward Euler Integrator
def int_eul(f, u_init, h, num):
    tvals = np.zeros(num)
    uvals = np.zeros((num, len(u_init)))
    uvals[0] = u_init
    
    for i in range(num - 1):
        t = tvals[i]
        u = uvals[i]
        
        tvals[i + 1] = t + h
        uvals[i + 1] = u + h * f(tvals, uvals, t, u)
        
    return (tvals, uvals)

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

#f_grav specific to Yoshida Integrator
def f_grav_yos(tvals, uvals, t, u):
    return -u * G * (u[0]**2 + u[1]**2) ** (-3/2)

#Yoshida Integrator
def int_yos(f, u0, v0, h, num):
    tvals = np.zeros(num)
    uvals = np.zeros((num, len(u0)))
    uvals[0] = u0
    
    #Coefficients from wikipedia
    cr2 = 2 ** (1/3)
    w0 = - cr2 / (2 - cr2)
    w1 = 1 / (2 - cr2)
    c1 = w1 / 2
    c2 = (w0 + w1) / 2
    
    for i in range(num - 1):
        t  = tvals[i]
        u0 = uvals[i]
        
        u1 = u0 + h * c1 * v0 
        v1 = v0 + h * w1 * f(tvals, uvals, t, u1)
        u2 = u1 + h * c2 * v1
        v2 = v1 + h * w0 * f(tvals, uvals, t, u2)
        u3 = u2 + h * c2 * v2
        v3 = v2 + h * w1 * f(tvals, uvals, t, u3)
        u4 = u3 + h * c1 * v3
        v0 = v3
        
        tvals[i + 1] = t + h
        uvals[i + 1] = u4
        
    return (tvals, uvals)

#Create plot
fig, ax = plt.subplots()
ax.plot(0,0, color='k', marker='o')

#Simulate using Forward Euler
(t, u_eul) = int_eul(f_grav, start, h, num)
point_eul = ax.plot(u_eul[0,0], u_eul[0,1], color='g', marker='o')[0]
trail_eul = ax.plot(u_eul[0,0], u_eul[0,1], color='g', label='Euler')[0]

#Simulate using Runge-Kutta
(t, u_rk4) = int_rk4(f_grav, start, h, num)
point_rk4 = ax.plot(u_rk4[0,0], u_rk4[0,1], color='b', marker='o')[0]
trail_rk4 = ax.plot(u_rk4[0,0], u_rk4[0,1], color='b', label='RK4')[0]

#Simulate using Yoshida
(t, u_yos) = int_yos(f_grav_yos, start[:2], start[2:], h, num)
point_yos = ax.plot(u_yos[0,0], u_yos[0,1], color='r', marker='o')[0]
trail_yos = ax.plot(u_yos[0,0], u_yos[0,1], color='r', label='Yoshida')[0]

#Set figure properties
ax.set(xlim=[-3, 7], ylim=[-5, 5])
ax.set_aspect('equal')
ax.set_title('Comparison of Integrators')
ax.legend()

#Animation frame update function
def update(frame):
    #Make the trails disappear over time
    #frame_start = max(0,frame - 60)
    frame_start = 0
    
    # update Euler data:
    point_eul.set_xdata(u_eul[frame : frame + 1, 0])
    point_eul.set_ydata(u_eul[frame : frame + 1, 1])
    trail_eul.set_xdata(u_eul[frame_start : frame, 0])
    trail_eul.set_ydata(u_eul[frame_start : frame, 1])
    
    # update RK4 data:
    point_rk4.set_xdata(u_rk4[frame : frame + 1, 0])
    point_rk4.set_ydata(u_rk4[frame : frame + 1, 1])
    trail_rk4.set_xdata(u_rk4[frame_start : frame, 0])
    trail_rk4.set_ydata(u_rk4[frame_start : frame, 1])
    
    # update Yoshida data:
    point_yos.set_xdata(u_yos[frame : frame + 1, 0])
    point_yos.set_ydata(u_yos[frame : frame + 1, 1])
    trail_yos.set_xdata(u_yos[frame_start : frame, 0])
    trail_yos.set_ydata(u_yos[frame_start : frame, 1])

#Run animation
ani = animation.FuncAnimation(fig=fig, func=update, frames=num, interval=20)
plt.show()
