import sympy
import numpy as np
import matplotlib.pyplot as plt

def rk45(f, t0, y0, t_bound, tol):

    # constants from John H. Mathews and Kurtis K. Fink 2004
    c2  = 1/4
    c3  = 3/8
    c32 = 3/32
    c33 = 9/32
    c4  = 12/13 
    c42 = 1932/2197
    c43 = -7200/2197 
    c44 = 7296/2197 
    c5  = 439/216
    c52 = -8 
    c53 = 3680/513
    c54 = - 845/4104
    c6  = 1/2
    c62 = -8/27
    c63 = 2
    c64 = -3544/2565
    c65 = 1859/4104
    c66 = -11/40

    # error term constants
    e41 = 25/216
    e42 = 1408/2565
    e43 = 2197/4101
    e44 = -1/5

    e51 = 16/135
    e52 = 6656/12825
    e53 = 28561/56430
    e54 = -9/50
    e55 = 2/55

    tk = t0
    yk = y0

    x_array = np.array([0])
    y_array = np.array([0])

    s = 1
    t = 0

    h = 0.1 

    while t < t_bound: 
        
        h = s * h 

        k1 = h * f(tk, yk) 
        k2 = h * f(tk + c2 * h, yk + c2 * k1)
        k3 = h * f(tk + c3 * h, yk + c32 * k1 + c33 * k2)
        k4 = h * f(tk + c4 * h, yk + c42 * k1 + c43 * k2 + c44 * k3)
        k5 = h * f(tk + h, yk + c5 * k1 - c52 * k2 + c53 * k3 + c54 * k4)
        k6 = h * f(tk + c6 * h, yk + c62 * k1 + c63 * k2 + c64 * k3 + c65 * k4 + c66 * k5)

        zk = yk + e51 * k1 + e52 * k3 + e53 * k4 + e54 * k5 + e55 * k6

        yk = yk + e41 * k1 + e42 * k3 + e43 * k4 + e44 * k5


        # update array's
        x_array = np.append(x_array, t)
        y_array = np.append(y_array, yk)

        # optimal step size scaller 
        s = (tol) / (2 * abs(zk - yk) ) ** 0.25     # maybe using np.linalg.norm() might be more efficent 

        # update t
        t = t + h # fixed
        
        print(h)
    return x_array, y_array

'''# Example usage
def func(t, y):
    k = 0.1 # added
    return -k * y # fixed

(x_vals , y_vals) = rk45(func, 0, 1, 1, 1e-5) # changed initial value of y

plt.scatter(x_vals, y_vals)
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Solution of the ODE dx/dt = -kx using the RKF45 method')
plt.grid(True)
plt.show()
'''

