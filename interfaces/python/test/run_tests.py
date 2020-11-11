import unittest
from ConexProgram import *
import numpy as np
import scipy.linalg as la

def randsym(d):
    A = np.matrix(np.random.randn(d, d)); 
    return np.matrix(0.5*np.add(A, A.transpose()))

def CheckErrors(sol):
    eps = 1e-5
    passes = sol.Ax_minus_b < eps and sol.x_dot_s < eps
    return passes

def log(name, val):
    print name +",", val," ",

def randominstance():
    n = 3
    m = 2
#    A = np.array(randn(n, m), order='F').astype(real)
    A = np.array(np.ones((n, m)), order='F').astype(real)
    A[0, 1] = 3
    A[1, 0] = 4

    b = np.array(np.random.randn(m)).astype(real)
    c = np.ones((n)).astype(real)
    b = np.array( (np.matrix(A).transpose()*np.matrix(c).transpose()).transpose()  ).astype(real)
    b = b[0,:]
    return A, b, c

def TestRandomInstance():
    prog = Conex()

    A1, b, c1 = randominstance()
    A2, b, c2 = randominstance()
    m = A1.shape[1]
    prog.AddLinearInequality(A1, c1)
    prog.AddLinearInequality(A2, c2)

    n = 4
    Amat = np.ones((n, n, m))
    cmat = np.eye(n, n)
    for i in range(0, m):
        Amat[:, :, i] = randsym(n)

    Amat[:, :, m - 1] = np.eye(n) * 0
    Amat[0, 0, m - 1] = 1

    prog.AddDenseLinearMatrixInequality(Amat, cmat)


    sol = prog.Maximize(b)
    return CheckErrors(sol.err)

def TestSparseInstance():
    num_vars = 3
    prog = Conex(num_vars)

    n = 4
    m1 = 2
    m2 = 2 
    Amat = np.ones((n, n, m1))
    cmat = np.eye(n, n)
    for i in range(0, m1):
        Amat[:, :, i] = randsym(n)

    prog.AddSparseLinearMatrixInequality(Amat, cmat, np.arange(0, m1))

    Amat = np.ones((n, n, m2))
    for i in range(0, m2):
        Amat[:, :, i] = randsym(n)

    prog.AddSparseLinearMatrixInequality(Amat, cmat, np.arange(1,  m2 + 1))

    sol = prog.Maximize(np.ones((prog.m, 1)))
    return sol.status == 1

def TestLMI():
    prog = Conex()

    n = 4
    m = 3
    Amat = np.ones((n, n, m))
    cmat = np.eye(n, n)

    if 0:
        Amat[:, :, 0] = np.zeros( (n, n)) 
        Amat[0, 0, 0] = 1
        Amat[0, 1, 0] = 1
        Amat[1, 0, 0] = 1

        Amat[:, :, 1] = np.zeros( (n, n)) 
        Amat[1, 1, 1] = 1
        Amat[0, 2, 1] = -1
        Amat[2, 0, 1] = -1

        Amat[:, :, 2] = np.zeros( (n, n)) 
        Amat[2, 2, 2] = 1
        Amat[0, 2, 2] = -1
        Amat[2, 0, 2] = -1
    else:
        for i in range(0, m):
            Amat[:, :, i] = randsym(n)
    
    Amat[:, :, m - 1] = np.eye(n) * 0
    Amat[0, 0, m - 1] = 1

    prog.AddDenseLinearMatrixInequality(Amat, cmat)

    b = prog.A[0].transpose().__mul__(np.eye(n))
    sol = prog.Maximize(b)
    return CheckErrors(sol.err) and sol.status

def TestBothInfeasLP():
    prog = Conex()

    m = 1;
    n = 3;
    A = randn(n, 1)
    c = np.array( (0, 1, -1))

    A[0] = 1;
    A[1] = 0;
    A[2] = 0;

    prog.AddLinearInequality(A, -c)
    b = np.ones(prog.m)

    prog.Maximize(-b)

def TestBlowUp():
    prog = Conex()

    m = 1;
    n = 2;
    A = randn(n, 1)
    c = np.array((0, 1))

    A[0] = 0;
    A[1] = 1;
    #y <= 1

    prog.AddLinearInequality(A, c)
    b = np.ones(prog.m)
    b[0] = 0

    y = prog.Maximize(b)
    print y

def DualInfeas():
    prog = Conex()

    m = 2;
    n = 2*m
    A = zeros(n, m)
    A[0:m, :] = eye(m)
    A[m:n, :] = eye(m)

    c = np.squeeze(np.ones((n, 1)))

    prog.AddLinearInequality(A, c)
    b = np.ones(prog.m)

    b[0] = 0
    b[1] = -1
    sol = prog.Maximize(b)
    return sol.status == 0 

def DualFailsSlater():
    prog = Conex()

    m = 2;
    n = m
    A = eye(m)
    c = np.squeeze(np.ones((n, 1)))

    prog.AddLinearInequality(A, c)
    b = np.ones(prog.m)

    b[0] = 1
    b[1] = 0
    sol = prog.Maximize(b)
    return CheckErrors(sol.err) and sol.status

def PrimalInfeas():
    prog = Conex()

    m = 2;
    n = 2*m
    A = zeros(n, m)
    A[0:m, :] = eye(m)
    A[m:n, :] = -eye(m)

    c = -np.squeeze(np.ones((n, 1)))

    prog.AddLinearInequality(A, c)
    b = np.ones(prog.m)

    b[0] = 1
    b[1] = 1
    sol = prog.Maximize(b)
    return sol.status == 0 



def VerifyMuIsNonIncreasing():
    prog = Conex()

    m = 2;
    n = 2*m
    A = zeros(n, m)
    A[0:m, :] = eye(m)
    A[m:n, :] = eye(m)

    c = -np.squeeze(np.ones((n, 1)))

    prog.AddLinearInequality(A, c)
    b = np.ones(prog.m)

    b[0] = 1
    b[1] = 1

    config = prog.DefaultConfiguration()
    config.max_iterations = 6
    sol = prog.Maximize(b, config)

    stats = prog.GetIterationNumberStats(-1)
    last_iteration_number = stats.iteration_number;
    
    if last_iteration_number + 1 > config.max_iterations:
        return False

    stats = prog.GetIterationStats()
    for i, stat in enumerate(stats):
        if i > 0:
            if stats[i].mu > stats[i - 1].mu:
                return False
    return True

def HermitianLMI():
    prog = Conex()
    try:
        prog.NewLinearMatrixInequality(2, 2);
        return True
    except:
        return False

    invalid_lmi_size = -2
    try:
        prog.NewLinearMatrixInequality(invalid_lmi_size, 2);
        return False
    except:
        return True


class UnitTests(unittest.TestCase):
    def test1(self):
        self.assertTrue(TestLMI())
    def test2(self):
        self.assertTrue(TestRandomInstance())
    def test3(self):
        self.assertTrue(DualFailsSlater())
    def test4(self):
        self.assertTrue(DualInfeas())
    def test5(self):
        self.assertTrue(PrimalInfeas())
    def test6(self):
        self.assertTrue(TestSparseInstance())
    def test7(self):
        self.assertTrue(VerifyMuIsNonIncreasing());
    def test8(self):
        self.assertTrue(HermitianLMI());
if __name__ == '__main__':
    unittest.main()
