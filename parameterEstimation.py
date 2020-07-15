#!/usr/bin/env python

"""
Parameter estimation for main.py script.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit

###############
# DEFINITIONS #
###############

def derivatives(init, t, beta, gamma, delta, zeta):
    """derivatives defines derivatives as a system of ordinary differential equations.

    Parameters:
        init (list): List object containing integer initial values for S,E,I,R,D.
        t (array): Numpy array object containing the time interval.
        beta (int): Integer value specifying the spreading rate.
        gamma (int): Integer value specifying the rate of people recovering.
        delta (int): Integer value specifying the incubation period.
        zeta (int): Integer value speciyfing the rate at which people die.

    Returns:
        dSdt,dEdt,dIdt,dRdt,dDdt (list): List object containing derivative arguments for odeint function.
    """
    S,E,I,R,D = init
    N = S + E + I + R + D
    dSdt = -beta * I * S/N
    dEdt = beta * I * S/N - delta * E
    dIdt = delta * E -  gamma * I - zeta * I
    dRdt = gamma * I
    dDdt = zeta * I
    return(dSdt,dEdt,dIdt,dRdt,dDdt)

def odeSolver(t, init, params):
    """Function to solve ordinary differential equation model.

    Parameters:
        t (array): Numpy array object containing the time interval.
        init (list): List object containing initial conditions.
        params (dictionary): Parameter object containing the model parameter.

    Returns:
        (array): Numpy array object containing the model solutions.
    """
    beta, gamma, delta, zeta = params["beta"].value, params["gamma"].value, params["delta"].value, params["zeta"].value
    return (odeint(derivatives, init, t, args=(beta, gamma, delta, zeta)))


def errorFUN(params, init, t, data):
    """Error function needed to conduct Levenberg–Marquardt minimization algorithm.

    Parameters:
        params (dictionary): Parameter object containing the model parameter.
        init (list): List object containing initial conditions.
        t (array): Numpy array object containing the time interval.
        data (dictionary): Dictionary object from Panda module containing covid-19 data.

    Returns:
        out (list): List object containing residuals.
    """
    data = data[["Active_infected", "Recovered", "Death"]].to_numpy(dtype=np.float)
    sol = odeSolver(t,init,params)
    out = (sol[:,2:5] - data).ravel()
    return (out)

def parameterOptimization(init, t, params, data):
    """Function to optimize given model parameters using the Levenberg-Marquardt minimization algorithm.

    Parameters:
        init (list): List object containing initial conditions.
        t (array): Numpy array object containing the time interval.
        params (dictionary): Parameter object containing the model parameter.
        data (dictionary): Dictionary object from Panda module containing covid-19 data.

    Returns:
        optimizedParams (dictionary): MinimizerResult object containing optimized model parameter and model statistics.

    """
    print("Optimizing model parameters...")
    optimizedParams = minimize(errorFUN, params, args=(init, t, data), method="leastsq")
    return (optimizedParams)

def simulation(data, inits, t_span, params):
    """Function to conduct full simulation on covid-19 data of Germany

    Parameters:
        data (dictionary): Dictionary object from Panda module containing covid-19 data.
        inits (list): List object containing initial conditions.
        t_span (array): Numpy array object containing arrays of time intervals.
        params (dictionary): Parameter object containing the model parameter.

    Returns:
        (array): Numpy array object containing fittedmath model solutions.
    """
    p = []
    fit = []
    simulated_data = []
    for i in range(len(t_span)):
        print("Start Iteration",i)

        p.append([params["beta"].value, params["gamma"].value, params["delta"].value, params["zeta"].value])
        init = inits[i]
        t = t_span[i]

        observed = data.iloc()[min(t):max(t)+1,:]
        optimized = parameterOptimization(init, t, params, observed)
        params = optimized.params

        simulated_data.append(odeSolver(np.append(t,[t[-1]+1]), init, params)[1:])
        inits.append(simulated_data[i][len(simulated_data[i])-1])
        fit.append([optimized.aic, optimized.bic])

    fit = [sum(i)/len(fit) for i in zip(*fit)]
    print("\nFit of the model:",
    "\n  Akaike information criterion: ", fit[0],
    "\n  Bayesian information criterion: ", fit[1])
    p = [sum(i)/len(p) for i in zip(*p)]
    print("\nMean of the optimized parameter values:",
    "\n  \u03B2: ", p[0], "\n  \u03B3: ", p[1],
    "\n  \u03B4: ", p[2], "\n  \u03B6: ", p[3],
    "\n")
    return(np.array(np.concatenate(simulated_data)), p[1:])
