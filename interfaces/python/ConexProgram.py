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
    shape = [0, 0]
    variables = []

    def __init__(self, x, variables = []):
        if len(variables) == 0:
            variables = [x.shape[2], range(0, x.shape[2])]
        if len(variables[1]) != x.shape[2]:
            raise NameError("Invalid LMI")

        self.matrices = x
        self.variables = variables[1]
        self.m = variables[0]
    
    def __mul__(self, x):
        if self.transposed:
            y = zeros(self.m, 1)
            for i, var in enumerate(self.variables):
                y[var] = np.trace(self.matrices[:, :, i]  * np.matrix(x))
            return y
        else:
            y = self.matrices[:, :, 0]  * 0 
            for i, var in enumerate(self.variables):
                y = y + self.matrices[:, :, i]  * float(x[var])
            return y

    def transpose(self):
        y = LMIOperator(self.matrices, [self.m, self.variables])
        y.transposed = not y.transposed
        return y;

class Conex:
    def __init__(self, m = -1):
        self.wrapper = conex
        self.a = self.wrapper.CONEX_CreateConeProgram();
        if (m >= 0):
            self.wrapper.CONEX_SetNumberOfVariables(self.a, m);
        self.num_constraints = 0
        self.num_lmi_constraints = 0
        self.linear_constraints = []
        self.A = []
        self.c = []
        self.m = m

    def __del__(self): 
        self.wrapper.CONEX_DeleteConeProgram(self.a);

    def GetIterationStats(self):
        stats = self.GetIterationNumberStats(-1)
        last_iteration_number = stats.iteration_number;

        stat_array = []
        for i in range(0, last_iteration_number + 1):
            stats = self.GetIterationNumberStats(i)
            stat_array.append(stats)

        return stat_array

    def GetIterationNumberStats(self, num):
        stats = self.wrapper.CONEX_IterationStats()
        self.wrapper.CONEX_GetIterationStats(self.a, stats, num)
        return stats

    def AddQuadraticCost(self, P): 
        if P.shape[0] != self.m or P.shape[1] != self.m:
            print self.m
            print P.shape[0]
            print P.shape[1]
            raise NameError("Cost matrix dimension does not match number of variables.")

        cost = self.wrapper.CONEX_AddQuadraticCost(self.a, P)
    def AddLinearInequality(self, A, c): 
        c_ = np.squeeze(np.array(c[:])).transpose()
        const_id = self.wrapper.CONEX_AddDenseLinearConstraint(self.a, A, c_)
        self.m = A.shape[1]
        self.n = A.shape[0]
        self.A.append(np.matrix(A))
        self.c.append(np.matrix(c))
        self.num_constraints = self.num_constraints + 1

    def AddLinearInequalities(self, A, lb, ub): 
        lb_ = np.squeeze(np.array(lb[:])).transpose()
        ub_ = np.squeeze(np.array(ub[:])).transpose()
        const_id = self.wrapper.CONEX_AddLinearInequalities(self.a, A, lb_, ub_)
        #TODO(FrankPermenter) Correctly build these matrices
        self.A.append(np.matrix(A))
        self.c.append(np.matrix(ub_))
        self.num_constraints = self.num_constraints + 1
    def DefaultConfiguration(self):
        config = self.wrapper.CONEX_SolverConfiguration()
        self.wrapper.CONEX_SetDefaultOptions(config);
        config.inv_sqrt_mu_max = 1000
        config.maximum_mu = 1e20
        config.max_iterations = 100
        config.final_centering_steps = 1
        config.prepare_dual_variables = 1
        config.infeasibility_threshold = 1e8
        config.divergence_upper_bound = 1
        return config


    def Solve(self, config = []): 
        if not config:
            config = self.DefaultConfiguration();

        config.enable_line_search = 1
        config.enable_rescaling = 0
        sol = Solution()
        sol.y = np.ones((self.m)).astype(real)
        sol.status = self.wrapper.CONEX_Solve(self.a,  config, sol.y)
        return sol


    def Maximize(self, b, config = []): 

        if not config:
            config = self.DefaultConfiguration();

        sol = Solution()
        b = np.matrix(b)
        if b.shape[1] > b.shape[0]:
            b = b.transpose()

        if b.shape[0] != self.m:
            raise NameError("Cost vector dimension does not match number of variables.")

        sol.y = np.ones((self.m)).astype(real)
        sol.status = self.wrapper.CONEX_Maximize(self.a, np.squeeze(np.array(b), 1), config, sol.y)

        return sol

    def GetDualVariables(self):
        x = []
        for i in range(0, self.num_constraints):
            c = np.matrix(self.c[i]).transpose()
            n2 = c.shape[0]
            n1 = c.shape[1]
            xi = np.ones((n1, n2)).astype(real)
            self.wrapper.CONEX_GetDualVariable(self.a, i, xi)
            xi = np.matrix(xi).transpose()
            x.append(xi)

        return x

    def NewLinearMatrixInequality(self, order, hyper_complex_dim):
        constraint = self.wrapper.intp();
        status = self.wrapper.CONEX_NewLinearMatrixInequality(self.a, order, hyper_complex_dim, constraint)
        if status != 0:
            raise NameError("Failed to add constraint.")
        self.num_constraints = self.num_constraints + 1
        self.c.append(np.zeros((order, order)).astype(real))
        return constraint.value()

    def NewLorentzConeConstraint(self, order):
        constraint = self.wrapper.intp();
        status = self.wrapper.CONEX_NewLorentzConeConstraint(self.a, order, constraint)
        if status != 0:
            raise NameError("Failed to add constraint.")
        self.num_constraints = self.num_constraints + 1
        return constraint.value()

    def NewLinearInequality(self, num_rows):
        constraint = self.wrapper.intp();
        status = self.wrapper.CONEX_NewLinearInequality(self.a, num_rows, constraint)
        if status != 0:
            raise NameError("Failed to add constraint.")
        self.num_constraints = self.num_constraints + 1
        return constraint.value()

    def NewQuadraticCost(self):
        constraint = self.wrapper.intp();
        status = self.wrapper.CONEX_NewQuadraticCost(self.a, constraint)
        if status != 0:
            raise NameError("Failed to create quadratic cost.")
        self.num_constraints = self.num_constraints + 1
        return constraint.value()

    def UpdateQuadraticCostMatrix(self, cost_id, value, row, col):
        status = self.wrapper.CONEX_UpdateQuadraticCostMatrix(self.a, cost_id, 
                float(value), int(row), int(col))
        if status != 0:
            raise NameError("Failed to update quadratic cost.")

    def UpdateLinearOperator(self, constraint, value, variable, row, col = 0, hyper_complex_dim = 0):
        status = self.wrapper.CONEX_UpdateLinearOperator(self.a, constraint,
                float(value), variable, row, col, hyper_complex_dim)
        if status != 0:
            raise NameError("Failed to update operator.")

    def UpdateAffineTerm(self, constraint, value,  row, col = 0, hyper_complex_dim = 0):
        status = self.wrapper.CONEX_UpdateAffineTerm(self.a, constraint, 
                float(value), row, col, hyper_complex_dim)
        if status != 0:
            raise NameError("Failed to update affine term.")

    def AddDenseLinearMatrixInequality(self, A, c): 
        self.n = A.shape[1]
        self.m = A.shape[2]
        self.A.append(LMIOperator(A))
        self.c.append(c)

        variables = np.arange(0, self.m)
        variables = variables.astype('int_')
        self.wrapper.CONEX_AddDenseLMIConstraint(self.a, A, c)
        self.num_constraints = self.num_constraints + 1

    def AddSparseLinearMatrixInequality(self, A, c, variables): 
        if np.max(variables) + 1 > self.m:
            print self.m
            raise NameError("Invalid sparse LMI." + str(self.m) + "!=" + str(np.max(variables+1)))
        self.A.append(LMIOperator(A, [self.m, variables]))
        self.c.append(c)

        variables = np.array(variables).astype('int_')
        self.wrapper.CONEX_AddSparseLMIConstraint(self.a, A, c, variables) 
        self.num_constraints = self.num_constraints + 1

    def ComputeErrors(self, y, xa, b):
        b = np.matrix(b)
        if b.shape[1] > b.shape[0]:
            b = b.transpose()

        err = Errors()
        sa = []
        for i in range(0, self.num_constraints):
            A = self.A[i]
            c = np.matrix(self.c[i]).transpose()
            n2 = c.shape[0]
            n1 = c.shape[1]

            x = xa[i]
            Ay = A.__mul__(y)
            if i == 0:
                Ax = A.transpose().__mul__(x)
            else:
                Ax = Ax + A.transpose().__mul__(x) 

            s = np.add(np.array(c) , np.array(-Ay))
            sa.append(s)

            if n2 == 1 or n1 == 1:
                err.x_dot_s = s.transpose() * x
                err.min_eig_S.append(min(s))
                err.min_eig_X.append(min(x))
            else:
                err.x_dot_s = err.x_dot_s + np.trace(np.matmul(s, x))
                err.min_eig_S.append(min(la.eig(s)[0]))
                err.min_eig_X.append(min(la.eig(x)[0]))

        err.Ax_minus_b = la.norm(b - Ax)
        return sa, err 
