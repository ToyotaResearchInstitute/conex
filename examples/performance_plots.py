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

def PlotMuUpdateVsDivBound():
    m = 30
    n = 30
    hyper_complex_dim = 1
    config1 = Conex().DefaultConfiguration()
    config2 = Conex().DefaultConfiguration()
    config3 = Conex().DefaultConfiguration()
    config4 = Conex().DefaultConfiguration()
    config1.inv_sqrt_mu_max = 100000
    config2.inv_sqrt_mu_max = 100000
    config3.inv_sqrt_mu_max = 100000
    config4.inv_sqrt_mu_max = 100000
    config1.final_centering_steps = 1
    config2.final_centering_steps = 1
    config3.final_centering_steps = 1
    config4.final_centering_steps = 1

    config1.divergence_upper_bound = .1
    config2.divergence_upper_bound = .5
    config3.divergence_upper_bound = 10
    config4.divergence_upper_bound = 1000

    configs = []
    labels = []
    configs.append(config1)
    configs.append(config2)
    configs.append(config3)
    configs.append(config4)
    hyper_complex_dim = 1
    mu = SolveRandomHermitianSDP(m, n, hyper_complex_dim, configs)

def SolveRandomHermitianSDP(num_variables, order, hyper_complex_dim, configs):
    for config in configs: 
        solution = prog.Maximize(b, config)
        mu = []
        mu = [stats.mu for stats in prog.GetIterationStats()]
        mus.append(mu)
        labels.append('divub ' + str(configs[i].divergence_upper_bound))

    Plot(title, show_plot, mus, labels)



def PlotMuUpdate(hyper_complex_dim, title, show_plot = False):
    m = 10
    config = Conex().DefaultConfiguration()
    config.divergence_upper_bound = 1000
    config.inv_sqrt_mu_max = 130000
    config.final_centering_steps = 1

    mu = []
    if 0:
        rank1 = 5
        rank2 = 10
        rank3 = 50
        rank4 = 100
        mu.append(SolveRandomSDP(m, rank1, config))
        mu.append(SolveRandomSDP(m, rank2, config))
        mu.append(SolveRandomSDP(m, rank3, config))
        mu.append(SolveRandomSDP(m, rank4, config))
    else:
        if hyper_complex_dim > 0:
            if hyper_complex_dim == 8:
                config.divergence_upper_bound = .1
                config.inv_sqrt_mu_max = 1000
            rank1 = 6; rank2 = 12; rank3 = 48; rank4 = 99;
            mu.append(SolveRandomHermitianSDP(m, rank1, hyper_complex_dim, config))
            mu.append(SolveRandomHermitianSDP(m, rank2, hyper_complex_dim, config))
            mu.append(SolveRandomHermitianSDP(m, rank3, hyper_complex_dim, config))
            mu.append(SolveRandomHermitianSDP(m, rank4, hyper_complex_dim, config))
        else:
            mu1, rank1 = SolveMixedConeProgram(m, 1, config)
            mu2, rank2 = SolveMixedConeProgram(m, 2, config)
            mu3, rank3 = SolveMixedConeProgram(m, 3, config)
            mu4, rank4 = SolveMixedConeProgram(m, 4, config) 
            mu.append(mu1)
            mu.append(mu2)
            mu.append(mu3)
            mu.append(mu4)

    labels = []
    labels.append("n = " + str(rank1));
    labels.append("n = " + str(rank2));
    labels.append("n = " + str(rank3));
    labels.append("n = " + str(rank4));
    print labels

    Plot(title, show_plot, mu, labels)

def Plot(filename, showplot, mu, labels):
    plt.clf()
    plt.rcParams.update({'font.size': 13})
    styles = ['k--',  'k-*', 'k-.', 'k-o', 'k-*']
    for i in range(0, len(mu)):
        mui = mu[i]
        plt.plot(range(1, 1+len(mui)),   np.log(mui),  styles[i], label=labels[i])

    plt.legend()
    plt.xlabel('Newton steps')
    plt.ylabel('log(mu)')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(filename + "mu_update.eps")
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

def AddRandomLorentzConeConstraint(self, numvars, order):
    constraint = self.NewLorentzConeConstraint(order)
    b = np.ones((numvars, 1)) * 0
    self.UpdateAffineTerm(constraint, 1, 0)

    for i in range(0, order + 1):
        for v in range(0, numvars):
            val = np.random.randn(1)[0]
            self.UpdateLinearOperator(constraint, val, v, i)
            if (i == 0):
                b[v] = b[v] + val
    return self, b

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

def SolveRandomHermitianSDP(num_variables, order, hyper_complex_dim, configs):
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

    if isinstance(configs, list):
        mus = []
        for config in configs: 
            solution = prog.Maximize(b, config)
            mu = []
            if solution.status:
                mu = [stats.mu for stats in prog.GetIterationStats()]
            mus.append(mu)
    else:
        solution = prog.Maximize(b, configs)
        mus = []
        if solution.status:
            mus = [stats.mu for stats in prog.GetIterationStats()]

    
    return mus

def SolveMixedConeProgram(num_variables, copies, config, hyper_complex_dim = [1, 2, 4]):
    prog = Conex(num_variables)
    order = 8
    b = np.zeros((num_variables, 1))
    rank = 0
    for i in range(0, copies):
        for hyper_complex_dim_i in hyper_complex_dim:
            prog, bi  = AddRandomLinearMatrixInequality(prog, num_variables, order, hyper_complex_dim_i)
            b = np.add(b, bi)
            rank = rank + order
        #prog, bi = AddRandomLorentzConeConstraint(prog, num_variables, order)
        #rank = rank + 2

    solution = prog.Maximize(b, config)
    mu = []
    if solution.status:
        mu = [stats.mu for stats in prog.GetIterationStats()]
    return mu, rank 
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

show_plot = True
#PlotGeodesicDistance(show_plot)
#PlotMuUpdate(1, "Real", show_plot)
#PlotMuUpdate(2, "Complex", show_plot)
#PlotMuUpdate(4, "Quaternion", show_plot)
#PlotMuUpdate(-1, "special", show_plot)
#PlotMuUpdate(8, "exceptional", show_plot)
#PlotMuUpdateVsDivBound()

