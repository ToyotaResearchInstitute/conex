import conex
import numpy as np
#print ezrange.range(3)
from myutils import *
import scipy.linalg as la
real = 'double'

def log(name, val):
    print name +",", val," ",

def randominstance():
    n = 3
    m = 2
    A = np.array(randn(n, m), order='F').astype(real)
    A = np.array(np.ones((n, m)), order='F').astype(real)
    A[0, 1] = 3
    A[1, 0] = 4

    b = np.array(np.random.randn(m)).astype(real)
    c = np.ones((n)).astype(real)
    b = np.array( (np.matrix(A).transpose()*np.matrix(c).transpose()).transpose()  ).astype(real)
    b = b[0,:]
    return A, b, c

class Solution:
    x = []
    y = []
    s = []
    status = []

class LMIOperator:
    matrices = []
    transposed = False
    m = 0

    def __init__(self, x):
        self.matrices = x
        self.m = x.shape[2]

    def __mul__(self, x):
        if self.transposed:
            y = zeros(self.m, 1)
            for i in range(0, self.m):
                y[i] = trace(self.matrices[:, :, i]  * np.matrix(x))
            return y
        else:
            y = self.matrices[:, :, 0]  * 0 
            for i in range(0, self.m):
                y = y + self.matrices[:, :, i]  * float(x[i])
            return y

    def transpose(self):
        y = LMIOperator(self.matrices)
        y.transposed = not y.transposed
        return y;

class Conex:
    def __init__(self):
        self.wrapper = conex
        self.a = self.wrapper.ConexCreateConeProgram();
        self.num_constraints = 0
        self.num_lmi_constraints = 0
        self.linear_constraints = []
        self.A = []
        self.c = []

    def __del__(self): 
        self.wrapper.ConexDeleteConeProgram(self.a);
        #print self.a
        #conex.destroy(self.a);

    def AddDenseLinearConstraint(self, A, c): 
        const_id = self.wrapper.ConexAddDenseLinearConstraint(self.a, A, c)
        self.m = A.shape[1]
        self.n = A.shape[0]
        self.A.append(np.matrix(A))
        self.c.append(np.matrix(c))
        self.num_constraints = self.num_constraints + 1

    def solve(self, b): 
        sol = Solution()
        b = np.matrix(b)
        if b.shape[1] > b.shape[0]:
            b = b.transpose()

        sol.y = np.ones((self.m)).astype(real)
        config = self.wrapper.ConexDefaultOptions();
        config.inv_sqrt_mu_max = 25000
        config.max_iter = 100
        config.final_centering_steps = 3
        config.prepare_dual_variables = 1
        config.dinf_limit = .6
        sol.status = self.wrapper.ConexSolve(self.a, np.squeeze(np.array(b)), config, sol.y)
        if sol.status.solved:
            sol.x, sol.s = self.get_slacks_and_dual_vars(np.matrix(sol.y).transpose(), b)
        return sol

    def addlmi(self, A, c): 
        self.n = A.shape[1]
        self.m = A.shape[2]
        self.A.append(LMIOperator(A))
        self.c.append(c)

        self.wrapper.ConexAddDenseLMIConstraint(self.a, A, c) 
        self.num_constraints = self.num_constraints + 1

    def get_slacks_and_dual_vars(self, y, b):
        xa = []
        sa = []
        print " "
        for i in range(0, self.num_constraints):
            A = self.A[i]
            c = np.matrix(self.c[i]).transpose()
            n2 = c.shape[0]
            n1 = c.shape[1]

            x = np.ones((n1, n2)).astype(real)
            self.wrapper.ConexGetDualVariable(self.a, i, x)
            x = np.matrix(x).transpose()
            Ay = A.__mul__(y)
            if i == 0:
                Ax = A.transpose().__mul__(x)
            else:
                Ax = Ax + A.transpose().__mul__(x) 

            s = np.add(np.array(c) , np.array(-Ay))

            #print s
            if n2 == 1 or n1 == 1:
                log("comp", s.transpose() * x)
                log("min x", min(x))
                log("min s", min(s))
            else:
                log("comp", np.trace(np.matmul(s, x)))
                log("minEigS", (min(la.eig(s)[0])))
                log("minEigX", (min(la.eig(x)[0])))
            xa.append(x)
            sa.append(s)

        log("resD", la.norm(np.matrix(b) - Ax))
        return xa, sa


def TestRandomInstance():
    prog = Conex()

    A1, b, c1 = randominstance()
    #A2, b, c2 = randominstance()
    m = A1.shape[1]
    prog.AddDenseLinearConstraint(A1, c1)
    #prog.AddDenseLinearConstraint(A2, c2)

    if 1:
        n = 4
        Amat = np.ones((n, n, m))
        cmat = np.eye(n, n)
        for i in range(0, m):
            Amat[:, :, i] = randsym(n)
    
        Amat[:, :, m - 1] = np.eye(n) * 0
        Amat[0, 0, m - 1] = 1

        prog.addlmi(Amat, cmat)

    prog.solve(b)
    
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

    prog.addlmi(Amat, cmat)

    b = prog.A[0].transpose().__mul__(np.eye(n))
    prog.solve(b)

def TestBothInfeasLP():
    prog = Conex()

    m = 1;
    n = 3;
    A = randn(n, 1)
    c = np.array( (0, 1, -1))

    A[0] = 1;
    A[1] = 0;
    A[2] = 0;

    prog.AddDenseLinearConstraint(A, -c)
    b = np.ones(prog.m)

    prog.solve(-b)

def TestBlowUp():
    prog = Conex()

    m = 1;
    n = 2;
    A = randn(n, 1)
    c = np.array((0, 1))

    A[0] = 0;
    A[1] = 1;
    #y <= 1

    prog.AddDenseLinearConstraint(A, c)
    b = np.ones(prog.m)
    b[0] = 0

    y = prog.solve(b)
    print y

def DualInfeas():
    prog = Conex()

    m = 2;
    n = 2*m
    A = zeros(n, m)
    A[0:m, :] = eye(m)
    A[m:n, :] = eye(m)

    c = np.squeeze(np.array(ones(n, 1)))

    prog.AddDenseLinearConstraint(A, c)
    b = np.ones(prog.m)

    b[0] = 0
    b[1] = -1
    sol = prog.solve(b)
    print "y", sol.y
    print "x", sol.x

def DualFailsSlater():
    prog = Conex()

    m = 2;
    n = m
    A = eye(m)
    c = np.squeeze(np.array(ones(n, 1)))

    prog.AddDenseLinearConstraint(A, c)
    b = np.ones(prog.m)

    b[0] = 1
    b[1] = 0
    sol = prog.solve(b)

def PrimalInfeas():
    prog = Conex()

    m = 2;
    n = 2*m
    A = zeros(n, m)
    A[0:m, :] = eye(m)
    A[m:n, :] = -eye(m)

    c = -np.squeeze(np.array(ones(n, 1)))

    prog.AddDenseLinearConstraint(A, c)
    b = np.ones(prog.m)

    b[0] = 1
    b[1] = 1
    sol = prog.solve(b)


#RunSolve()
#TestClass()
TestLMI()
#TestBothInfeasLP()
#DualFailsSlater()
#DualInfeas()
#PrimalInfeas()
TestRandomInstance()
