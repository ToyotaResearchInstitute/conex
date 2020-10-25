classdef ConexAPI
  
 methods
  function self = ConexAPI(self)
    if ~libisloaded('libconex')
      loadlibrary libconex.so conex.h
    end
  end

  function RunTests(self)

    num_var = 2;
    num_constraint = 4;

    A = randn(num_constraint, num_var);
    c = ones(num_constraint, 1); 
    cinfeasible = -3*ones(num_constraint, 1); 

    b = A' * ones(num_constraint, 1);
    if ~self.Solve(A, b, c)
      error('Test failed: expected feasible solution found')
    end
    if self.Solve(A, b, cinfeasible)
      error('Test failed: expected infeasibility')
    end
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
