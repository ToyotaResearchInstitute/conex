from ConexProgram import *
import numpy as numpy
import matplotlib.pyplot as plt
import sys
import threading
import os
from matplotlib.ticker import MaxNLocator
from myutils import *

def geodistv(x, y):
    return norm2(logm( sqrtm(x) * inv(y) * sqrtm(x)))

def geodist(v0, varray):
    n = v0.shape[0]
    val = [geodistv(v0, v) for v in varray]
    return val



def PlotMuUpdate():
    m = 10
    eps = 0.005

    config = Conex().DefaultConfiguration()
    config.divergence_upper_bound = 100
    config.inv_sqrt_mu_max = 100000

    mu5 = SolveRandomSDP(3, 5, config)
    mu25 = SolveRandomSDP(3, 10, config)
    mu50 = SolveRandomSDP(3, 50, config)
    mu100 = SolveRandomSDP(3, 100, config)

    plt.rcParams.update({'font.size': 13})
    plt.plot( range(1, 1+len(mu5)),   np.log(mu5),  'k--', label="n=5")
    plt.plot(range(1, 1+len(mu25)),  np.log(mu25),  'k-+', label="n=25")
    plt.plot(range(1, 1+len(mu50)), np.log(mu50),  'k-o', label="n=50")
    plt.plot(range(1, 1+len(mu100)), np.log(mu100),  'k', label="n=100")
    plt.legend()
    plt.xlabel('Newton steps (long)')
    plt.ylabel('log(mu)')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show(True)

def random_symmetric_matrix(n):
    x = np.matrix(np.random.randn(n, n))
    x = x + x.transpose() 
    return x

def CenteringConfig(mu):
    config = Conex().DefaultConfiguration()
    config.minimum_mu = 1
    config.maximum_mu = 1
    config.inv_sqrt_mu_max = 1
    config.max_iterations = 1
    config.divergence_upper_bound = 1
    config.centering_tolerance = 30
    config.minimum_mu = mu
    config.maximum_mu = mu
    config.inv_sqrt_mu_max = 1.0/np.sqrt(mu)
    return config

def GetCenteringIterates(A, b, c, mu):
    prog = Conex()
    prog.AddDenseLinearMatrixInequality(A, c)

    config = CenteringConfig(mu)
    config.initialization_mode = 0
    config.prepare_dual_variables = 0
    config.max_iterations = 1;
    warray = []
    for j in range(0, 16):
        solution = prog.Maximize(b, config)
        warray.append(solution.x[0])
        config.initialization_mode = 1
    return warray

def GetGeodesicDistanceToMuCenteredPoint():
    num_experiments = 10
    num_variables = 10
    n = 10
    mu = .01

    A = np.ones((n, n, num_variables))
    for i in range(0, num_variables):
        A[:, :, i] = random_symmetric_matrix(n)

    for i in range(0, num_experiments):
        warray = []
        w0 = np.eye(n, n)
        v = np.random.randn(n, 1)
        v = v / la.norm(v) 
        v = v * (1.3*i+0.01)*.45
        for j in range(0, n):
            w0[j, j] = np.exp(v[j])

        c = np.sqrt(mu) * inv(w0)
        constraint_operator = LMIOperator(A)
        b = constraint_operator.transpose() * (w0 * np.sqrt(mu))
        xarray = GetCenteringIterates(A, b, c, mu)

        metric = geodist(w0 * np.sqrt(mu), xarray)
        if i == 0:
            div = [metric] 
        else:
            div.append(metric)

    return div

def PlotGeodesicDistance():
    div = GetGeodesicDistanceToMuCenteredPoint()
    logscale = False
    for i in range(0, 2):
        plt.clf()
        plt.rcParams.update({'font.size': 13})

        for d in div:
            d = np.real(d)
            if logscale:
                print d
                plt.plot(np.log(np.abs(d)))
            else: 
                plt.plot(d)

        plt.xlabel('Newton step')
        plt.ylabel('Geodesic Distance')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        if logscale:
            plt.savefig("converge_log.eps")
        else:

            plt.savefig("converge.eps")
        logscale = True
        plt.show()

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
    config.initialization_mode = 0
    return [stats.mu for stats in prog.GetIterationStats()]

PlotMuUpdate()
#PlotGeodesicDistance()
