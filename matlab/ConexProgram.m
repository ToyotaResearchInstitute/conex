classdef ConexProgram
  properties(Access=private)
    p
  end
  
  methods
    function self = ConexProgram(self)
      if ~libisloaded('libconex')
        loadlibrary libconex.so conex.h
      end
      self.p = calllib('libconex','ConexCreateConeProgram');
    end

    function delete(self)
      calllib('libconex','ConexDeleteConeProgram', self.p);
    end

    function AddLinearInequality(self, A, c)
      num_var = size(A, 2);
      num_constraint = size(A, 1);
      Aptr = libpointer('doublePtr', full(A(:)));
      Cptr = libpointer('doublePtr', full(c));
      calllib('libconex', 'ConexAddDenseLinearConstraint', self.p, Aptr,  ...
      num_constraint, num_var, Cptr, num_constraint);
    end

    function AddLinearMatrixInequality(self, A, c)
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

      Aptr = libpointer('doublePtr', full(A(:)));
      Cptr = libpointer('doublePtr', full(c));
      calllib('libconex', 'ConexAddDenseLMIConstraint', self.p, Aptr,  n, n, m, Cptr, n, n);
    end

    function [y, status] = Maximize(self, b)
        if size(b, 2) > 1 && size(b, 1) > 1
          error('Cost must be a vector.')
        end
        num_var = length(b);
        bptr = libpointer('doublePtr', full(b));
        yptr = libpointer('doublePtr', zeros(num_var, 1));
        status = calllib('libconex', 'ConexSolve', self.p, bptr, length(b), yptr, num_var);
        y = yptr.Value;
    end

    function solved = Solve(self, A, b, c)
      if (size(b, 1) ~= size(A, 2))
        error('Invalid input dimensions.')
      end

      if (size(c, 1) ~= size(A, 1))
        error('Invalid input dimensions.')
      end

      try 
        p = calllib('libconex','ConexCreateConeProgram');
        num_var = size(A, 2);
        num_constraint = size(A, 1);

        b = A' * ones(num_constraint, 1);

        Aptr = libpointer('doublePtr', A(:));
        Cptr = libpointer('doublePtr', c);
        bptr = libpointer('doublePtr', b);
        calllib('libconex', 'ConexAddDenseLinearConstraint', p, Aptr, num_constraint, num_var, Cptr, num_constraint);


        yptr = libpointer('doublePtr', zeros(num_var, 1));
        solved = calllib('libconex', 'ConexSolve', p, bptr, length(b), yptr, num_var);
      catch
        calllib('libconex','ConexDeleteConeProgram', p);
      end

      calllib('libconex','ConexDeleteConeProgram', p);
    end
 end
end
