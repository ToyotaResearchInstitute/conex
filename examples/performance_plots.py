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

def PlotMuUpdate(hyper_complex_dim, title, show_plot = False):
    m = 10

    config = Conex().DefaultConfiguration()
    config.divergence_upper_bound = 1000
    config.inv_sqrt_mu_max = 130000
    config.final_centering_steps = 3

    if 0:
        mu5 = SolveRandomSDP(m, 5, config)
        mu25 = SolveRandomSDP(m, 10, config)
        mu50 = SolveRandomSDP(m, 50, config)
        mu100 = SolveRandomSDP(m, 100, config)
    else:
        mu5 = SolveRandomHermitianSDP(m, 5, hyper_complex_dim, config)
        mu25 = SolveRandomHermitianSDP(m, 10, hyper_complex_dim, config)
        mu50 = SolveRandomHermitianSDP(m, 50, hyper_complex_dim, config)
        mu100 = SolveRandomHermitianSDP(m, 100, hyper_complex_dim, config)

    return
    plt.rcParams.update({'font.size': 13})
    plt.plot( range(1, 1+len(mu5)),   np.log(mu5),  'k--', label="n=5")
    plt.plot(range(1, 1+len(mu25)),  np.log(mu25),  'k-+', label="n=25")
    plt.plot(range(1, 1+len(mu50)), np.log(mu50),  'k-o', label="n=50")
    plt.plot(range(1, 1+len(mu100)), np.log(mu100),  'k', label="n=100")
    plt.legend()
    plt.xlabel('Newton steps (long)')
    plt.ylabel('log(mu)')
    plt.title(title)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(title + "mu_update.eps")
    if show_plot:
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
        x = prog.GetDualVariables()
        warray.append(x[0])
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

def PlotGeodesicDistance(show_plot):
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
        if show_plot:
            plt.show()


def AddRandomLinearMatrixInequality(self, numvars, order, hyper_complex_dim):
    constraint = self.NewLinearMatrixInequality(order, hyper_complex_dim)
    b = np.ones((numvars, 1)) * 0
    for k in range(0, hyper_complex_dim):
        for i in range(0, order):
            jstart = i
            self.UpdateAffineTerm(constraint, 1, i, i, 0)
            if k > 0:
                jstart = i + 1
            for j in range(jstart, order):
                for v in range(0, numvars):
                    val = np.random.randn(1)[0]
                    self.UpdateLinearOperator(constraint, val, v, i, j, k)
                    if (i == j) and k == 0:
                        b[v] = b[v] + val
    return self, b

def SolveRandomHermitianSDP(num_variables, order, hyper_complex_dim, config):
    prog = Conex(num_variables)

    if hyper_complex_dim != 8:
        prog, b  = AddRandomLinearMatrixInequality(prog, num_variables, order, hyper_complex_dim)
    else:
        for i in range(0, order / 3):
            prog, bi  = AddRandomLinearMatrixInequality(prog, num_variables, 3, hyper_complex_dim)
            if i == 0:
                b = bi
            else:
                b = np.add(b, bi)

    solution = prog.Maximize(b, config)
    mu = []
    if solution.status:
        mu = [stats.mu for stats in prog.GetIterationStats()]
    return mu 

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

show_plot = False
PlotGeodesicDistance(show_plot)
PlotMuUpdate(1, "Real", show_plot)
PlotMuUpdate(2, "Complex", show_plot)
PlotMuUpdate(4, "Quaternion", show_plot)
#PlotMuUpdate(8, "Octonion")
