classdef ConexProgram < handle
  properties(Access=private)
    p
    constraints
  end

  properties(Access=public)
    options
  end

 methods(Static)
    function [m, n] = VerifyLMIData(A, c)
      if size(c, 2) ~= size(c, 1)
        error('Affine term c must be square matrix.')
      end
      n = size(c, 1);
      if mod(size(A, 2), n) ~= 0
        error('LMI matrices have incompatible dimension')
      end
      if size(A, 1) ~= n
        error('LMI matrices have incompatible dimension')
      end
      m = size(A, 2) / n;
    end
 end

 methods
    function self = ConexProgram(num_vars)
      if ~libisloaded('libconex')
        loadlibrary libconex.so conex.h
      end
      self.p = calllib('libconex', 'CONEX_CreateConeProgram');
      calllib('libconex', 'CONEX_SetNumberOfVariables', self.p, num_vars); 
      self.options = libstruct('CONEX_SolverConfiguration');
      % Force matlab to allocate memory for options
      self.options.divergence_upper_bound = 1;
      calllib('libconex', 'CONEX_SetDefaultOptions', self.options);
      self.options.prepare_dual_variables = 1;
    end

    function delete(self)
      calllib('libconex', 'CONEX_DeleteConeProgram', self.p);
    end

    function AddLinearInequality(self, A, c)
      num_var = size(A, 2);
      num_constraint = size(A, 1);
      Aptr = libpointer('doublePtr', full(A(:)));
      Cptr = libpointer('doublePtr', full(c));
      self.constraints(end+1) = calllib('libconex', 'CONEX_AddDenseLinearConstraint', self.p, Aptr,  ...
      num_constraint, num_var, Cptr, num_constraint);
    end

    function AddLinearMatrixInequality(self, A, c, variables)
      if nargin < 4
        self.AddDenseLinearMatrixInequality(A, c)
      else
        self.AddSparseLinearMatrixInequality(A, c, variables)
      end
    end

    function x = GetDualVariable(self, i)
       dual_var_size = calllib('libconex', 'CONEX_GetDualVariableSize', self.p, i);
       xptr = libpointer('doublePtr', zeros(dual_var_size, 1));
       dual_var_size = calllib('libconex', 'CONEX_GetDualVariable', self.p, i, xptr, dual_var_size, 1);
       x = xptr.Value;
    end

    function AddDenseLinearMatrixInequality(self, A, c)
      [m, n] = ConexProgram.VerifyLMIData(A, c);

      Aptr = libpointer('doublePtr', full(A(:)));
      Cptr = libpointer('doublePtr', full(c));
      self.constraints(end + 1) = calllib('libconex', 'CONEX_AddDenseLMIConstraint', self.p, Aptr,  n, n, m, Cptr, n, n);
    end

    function AddSparseLinearMatrixInequality(self, A, c, vars)
      [m, n] = ConexProgram.VerifyLMIData(A, c);

      Aptr = libpointer('doublePtr', full(A(:)));
      Cptr = libpointer('doublePtr', full(c));
      varPtr = libpointer('longPtr', full(vars));
      self.constraints(end + 1) = calllib('libconex', 'CONEX_AddSparseLMIConstraint', self.p, Aptr,  n, n, m, Cptr, n, n, varPtr, m);
    end

    function [y, x, status] = Maximize(self, b)
      if size(b, 2) > 1 && size(b, 1) > 1
        error('Cost must be a vector.')
      end
      num_var = length(b);
      bptr = libpointer('doublePtr', full(b));
      yptr = libpointer('doublePtr', zeros(num_var, 1));
      status = calllib('libconex', 'CONEX_Maximize', self.p, bptr, length(b), self.options, yptr, num_var);

      x = {};
      for i = 1:length(self.constraints)
        x{i} = self.GetDualVariable(self.constraints(i));
      end
      y = yptr.Value;
    end

 end
end
