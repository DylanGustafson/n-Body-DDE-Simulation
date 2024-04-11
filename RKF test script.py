import numpy as np
import matplotlib.pyplot as plt


# RKF function
# f    function 
# a    start point
# b    end point
# x0   initial value 
# tol  tolernce for local truncation error
# hmax max time step
# hmin min time step  

def rkf(f, a, b, x0, tol, hmax, hmin):

    # constants from rkf declared as floats
    a2  = 2.500000000000000e-01  # 1/4
    a3  = 3.750000000000000e-01  # 3/8
    a4  = 9.230769230769231e-01  # 12/13
    a5  = 1.000000000000000e+00  # 1
    a6  = 5.000000000000000e-01  # 1/2

    b21 = 2.500000000000000e-01  # 1/4
    b31 = 9.375000000000000e-02  # 3/32
    b32 = 2.812500000000000e-01  # 9/32
    b41 = 8.793809740555303e-01  # 1932/2197
    b42 =-3.277196176604461e+00  # -7200/2197
    b43 = 3.320892125625853e+00  # 7296/2197
    b51 = 2.032407407407407e+00  # 439/216
    b52 =-8.000000000000000e+00  # -8
    b53 = 7.173489278752436e+00  # 3680/513
    b54 =-2.058966861598441e-01  # -845/4104
    b61 =-2.962962962962963e-01  # -8/27
    b62 = 2.000000000000000e+00  # 2
    b63 =-1.381676413255361e+00  # -3544/2565
    b64 = 4.529727095516569e-01  # 1859/4104
    b65 =-2.750000000000000e-01  # -11/40

    r1  = 2.777777777777778e-03  # 1/360
    r3  =-2.994152046783626e-02  # -128/4275
    r4  =-2.919989367357789e-02  # -2197/75240
    r5  = 2.000000000000000e-02  # 1/50
    r6  = 3.636363636363636e-02  # 2/55

    c1  = 1.157407407407407e-01  # 25/216
    c3  = 5.489278752436647e-01  # 1408/2565
    c4  = 5.353313840155945e-01  # 2197/4104
    c5  =-2.000000000000000e-01  # -1/5

    # initaliz varibles 
    current_time = a # current time (set to beginning of function)
    x = x0 # current value at solution (set to inital value)
    h = hmax # start a max step size 

    T = np.array([current_time]) # array to store points in time
    X = np.array([x]) # array to store solutions

    # RKF main loop 
    while current_time < b: # run untill reach end 

        if current_time + h > b:
            h = b - current_time

        # calculate intermittent value
        k1 = h * f(current_time, x)
        k2 = h * f(current_time + a2 * h, x + b21 * k1)
        k3 = h * f(current_time + a3 * h, x + b31 * k1 + b32 * k2)
        k4 = h * f(current_time + a4 * h, x + b41 * k1 + b42 * k2 + b43 * k3)
        k5 = h * f(current_time + a5 * h, x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
        k6 = h * f(current_time + a6 * h, x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)

        # error estimate 
        error = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / h

        # if error ok 
        if error <= tol:
            current_time = current_time + h # move forward 
            x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 # calc wheighted values 
            T = np.append(T, current_time)
            X = np.append(X, x)

        # adjust step size based on error 
        h = h * min(max(0.84 * (tol / error) ** 0.25, 0.1), 4.0)
        
        # make sure step size is not too small 
        if h > hmax:
            h = hmax
        elif h < hmin:
            raise RuntimeError("Minimum step size reached.")

    # return solution arrays 
    return T, X


# Define the ODE
def f(current_time, x):
    k = 0.5  # decay rate
    return -k * x

# Set the initial conditions and parameters
a = 0  # start time
b = 10  # end time
x0 = 1  # initial condition
tol = 1e-6  # tolerance
hmax = 0.5  # maximum step size
hmin = 1e-6  # minimum step size

# Solve the ODE using the RKF45 method
T, X = rkf(f, a, b, x0, tol, hmax, hmin)

# Plot the solution
plt.scatter(T, X)
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Solution of the ODE dx/dt = -kx using the RKF45 method')
plt.grid(True)
plt.show()