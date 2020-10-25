

%loadlibrary codegen_solver.so codegen_solver.h
%
%param_dim = calllib('codegen_solver','parameter_dim');
%solution_dim = calllib('codegen_solver','solution_dim');
%
%y = libpointer('doublePtrPtr', zeros(solution_dim, 1));
%param = libpointer('doublePtrPtr', zeros(param_dim, 1));
%
%calllib('codegen_solver','set_default_parameters', param);
%calllib('codegen_solver','call_solver', param.Value, y);
%
%display('')
%display('Snippet of solution:')
%y.Value(1:10)

loadlibrary libconex.so ../swig/conex.h
p = calllib('libconex','ConexCreateConeProgram');
calllib('libconex','ConexDeleteConeProgram', p);

