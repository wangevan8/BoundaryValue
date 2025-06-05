import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Parameters
h = 2/200  # Step size
x = np.arange(0, 2.01, h)  # Domain from 0 to 2 inclusive

# Shooting method
def ode_system(t, y):
    """
    Converts the second-order ODE y'' + x*y = 0 into a first-order system.

    Parameters:
    t (float): The independent variable.
    y (list or array): A 2-element list or array where y[0] = y, y[1] = y'.

    Returns:
    list: A list [y', y''] representing the derivatives.
    """
    y1, y2 = y
    return [y2, -t * y1]

def shooting_method(s_guess):
    """
    Solves the ODE IVP using the shooting method for a given initial slope guess.

    Parameters:
    s_guess (float): The guessed value for y'(0).

    Returns:
    ndarray: The computed y values at each point in x over the interval [0, 2].
    """
    # Initial conditions: y(0) = 1, y'(0) = s_guess
    sol = solve_ivp(ode_system, [0, 2], [1, s_guess], method='RK45', 
                    dense_output=True, rtol=1e-8, atol=1e-8)
    
    # Extract solution at all x values
    solution = sol.sol(x)
    return solution[0, :]  # Return y values

def objective_function(s):
    """
    Objective function for root finding. Computes the difference y(2) - 2.

    Parameters:
    s (float): The initial guess for y'(0).

    Returns:
    float: The value of y(2) - 2, which should be zero at the correct initial slope.
    """
    solution = shooting_method(s)
    return solution[-1] - 2  # y(2) - 2

# Find the correct initial slope s that makes y(2) = 2 (brent)
result = root_scalar(objective_function, bracket=[0, 5], method='brentq')
s_optimal = result.root
print(f"Optimal initial slope y'(0) = {s_optimal:.6f}")
y_shooting = shooting_method(s_optimal)

# Results table
print("\nSolution Table:")
print("    x    |    y(x)   ")
print("---------+-----------")
for i in range(len(x)):
    if i % 1 == 0:  
        print(f"{x[i]:7.2f} | {y_shooting[i]:10.6f}")

