import conex
import numpy as np
import scipy.linalg as la

real = 'double'
def zeros(n, m):
    return np.matrix(np.zeros((n, m)))

def eye(n):
    return np.matrix(np.eye(n, n))


class Errors:
    Ax_minus_b = 0
    x_dot_s = 0
    min_eig_S = []
    min_eig_X = []

class Solution:
    err = Errors()
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
                y[i] = np.trace(self.matrices[:, :, i]  * np.matrix(x))
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
        config = self.wrapper.ConexSolverConfiguration()
        config.inv_sqrt_mu_max = 25000
        config.max_iterations = 100
        config.final_centering_steps = 3
        config.prepare_dual_variables = 1
        config.infeasibility_threshold = 1e8
        config.divergence_upper_bound = 1
        sol.status = self.wrapper.ConexSolve(self.a, np.squeeze(np.array(b)), config, sol.y)
        if sol.status:
            sol.x, sol.s, sol.err = self.get_slacks_and_dual_vars(np.matrix(sol.y).transpose(), b)
        return sol

    def addlmi(self, A, c): 
        self.n = A.shape[1]
        self.m = A.shape[2]
        self.A.append(LMIOperator(A))
        self.c.append(c)

        self.wrapper.ConexAddDenseLMIConstraint(self.a, A, c) 
        self.num_constraints = self.num_constraints + 1

    def get_slacks_and_dual_vars(self, y, b):
        err = Errors()
        xa = []
        sa = []
        for i in range(0, self.num_constraints):
            A = self.A[i]
            c = np.matrix(self.c[i]).transpose()
            n2 = c.shape[0]
            n1 = c.shape[1]

            x = np.ones((n1, n2)).astype(real)
            self.wrapper.ConexGetDualVariable(self.a, i, x)
            x = np.matrix(x).transpose()
            xa.append(x)
            Ay = A.__mul__(y)
            if i == 0:
                Ax = A.transpose().__mul__(x)
            else:
                Ax = Ax + A.transpose().__mul__(x) 

            s = np.add(np.array(c) , np.array(-Ay))

            if n2 == 1 or n1 == 1:
                err.x_dot_s = s.transpose() * x
                err.min_eig_S.append(min(s))
                err.min_eig_X.append(min(x))
            else:
                err.x_dot_s = err.x_dot_s + np.trace(np.matmul(s, x))
                err.min_eig_S.append(min(la.eig(s)[0]))
                err.min_eig_X.append(min(la.eig(x)[0]))

        err.Ax_minus_b = la.norm(b - Ax)
        return xa, sa, err 
