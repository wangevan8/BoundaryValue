# Code referenced from docs.scipy.org
# Result table verified and generated using Claude.ai
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Parameters
h = 2/200  # Step size
x = np.arange(0, 2.01, h)  # Domain from 0 to 2 inclusive

# Shooting method
def ode_system(t, y):
    # Define the ODE system y'' + xy = 0 as a first-order system
    y1, y2 = y
    return [y2, -t * y1]

def shooting_method(s_guess):
    # Solve the IVP with initial guess for y'(0)
    # Initial conditions: y(0) = 1, y'(0) = s_guess
    sol = solve_ivp(ode_system, [0, 2], [1, s_guess], method='RK45', 
                    dense_output=True, rtol=1e-8, atol=1e-8)
    
    # Extract solution at all x values
    solution = sol.sol(x)
    return solution[0, :]  # Return y values

def objective_function(s):
    # Define objective function for root finding: y(2) - 2 = 0
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

