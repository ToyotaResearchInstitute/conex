from ConexProgram import *
import numpy as numpy
import matplotlib.pyplot as plt
import sys
import threading
import os
from matplotlib.ticker import MaxNLocator

def MuUpdateConfig():
    config = Conex().DefaultConfiguration()
    config.max_iterations = 30
    config.divergence_upper_bound = 10 
    config.final_centering_steps = 1 
    return config

def PlotMuUpdate():
    m = 10
    kf = 25000
    eps = 0.005

    config = Conex().DefaultConfiguration()
    config.divergence_upper_bound = 100

    mulong5 = SolveRandomSDP(3, 5, config)
    mulong25 = SolveRandomSDP(3, 10, config)
    mulong50 = SolveRandomSDP(3, 50, config)
    mulong100 = SolveRandomSDP(3, 100, config)

    plt.rcParams.update({'font.size': 13})
    plt.plot( range(1, 1+len(mulong5)),   np.log(mulong5),  'k--', label="n=5")
    plt.plot(range(1, 1+len(mulong25)),  np.log(mulong25),  'k-+', label="n=25")
    plt.plot(range(1, 1+len(mulong50)), np.log(mulong50),  'k-o', label="n=50")
    plt.plot(range(1, 1+len(mulong100)), np.log(mulong100),  'k', label="n=100")
    plt.legend()
    plt.xlabel('Newton steps (long)')
    plt.ylabel('log(mu)')
 #   plt.title('Short vs long step: Newton steps')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

#    plt.savefig("mupdate.eps")
    plt.show(True)

def random_symmetric_matrix(n):
    x = np.matrix(np.random.randn(n, n))
    x = x + x.transpose() 
    return x

def CenteringConfig():
    config = Conex().DefaultConfiguration()
    config.minimum_mu = 1
    config.maximum_mu = 1
    config.inv_sqrt_mu_max = 1
    config.max_iterations = 1
    config.divergence_upper_bound = 1
    config.final_centering_steps = 0
    config.centering_tolerance = 30
    return config

def Centering():
    x = []
    num_variables = 10
    n = 40
    config = CenteringConfig()
    w0 = np.eye(n, n)
    v = np.random.randn(n, 1) * 1.1
    for i in range(0, n):
        w0[i, i] = np.exp(v[i])


    prog = Conex()
    A = np.ones((n, n, num_variables))
    for i in range(0, num_variables):
        A[:, :, i] = random_symmetric_matrix(n)
    if len(w0) == 0:
        w0 = np.eye(n, n)

    c = w0
    constraint_operator = LMIOperator(A)
    b = constraint_operator.transpose() * la.inv(w0)

    prog.AddDenseLinearMatrixInequality(A, c)
    config_def = Conex().DefaultConfiguration()
    config_def.max_iterations = 100
    config_def.minimum_mu = 1
    config_def.maximum_mu = 1
    config_def.inv_sqrt_mu_max = 1
    solution = prog.Maximize(b, config_def)

    config.initialization_mode = 0
    config.prepare_dual_variables = 0
    for i in range(0, 10):
        solution = prog.Maximize(b, config)
        x.append(solution.x)
        config.initialization_mode = 1


def SolveRandomSDP(num_variables, n, config, w0 = []):
    prog = Conex()
    A = np.ones((n, n, num_variables))
    for i in range(0, num_variables):
        A[:, :, i] = random_symmetric_matrix(n)
    if len(w0) == 0:
        w0 = np.eye(n, n)

    c = w0
    constraint_operator = LMIOperator(A)
    b = constraint_operator.transpose() * la.inv(w0)
    prog.AddDenseLinearMatrixInequality(A, c)
    solution = prog.Maximize(b, config)
    config.initialization_mode = 1
    return [stats.mu for stats in prog.GetIterationStats()]

#PlotMuUpdate()
Centering()
