%Calls conex using SeDuMI formatted inputs and outputs.
function [x, y, info] = conex(Ain, bin, c, K, pars)
[A, b, T] = CleanLinear(Ain, bin);

cone = coneBase(K);
A = cone.Symmetrize(A); c = cone.Symmetrize(c');
if IsNontrivialField(K, 'l') 
  error('Cone not supported yet');
end
if IsNontrivialField(K, 'r') 
  error('Cone not supported yet');
end
if IsNontrivialField(K, 'q') 
  error('Cone not supported yet');
end

if nargin < 5
  pars =[];
  pars.errors = 0;
  pars.blkdiag = length(K.s) > 1;
end

info.numerr = 0;
info.pinf = 0;
info.dinf = 0;
info.feasratio = 1;
info.timing = [0, 0, 0]; 
info.cpusec = 0;

if (pars.blkdiag)
  problem = ConexPreprocess(A, b, c, K);
else
  problem.K = K;
  problem.b = b;
end

p = ConexProgram(length(b));
if length(K.s)  > 1
  for i = 1:length(problem.K.s)
    n = problem.K.s(i);
    Ai2 = problem.constraints{i}.matrix_conex_format;
    p.AddSparseLinearMatrixInequality(Ai2, reshape(problem.constraints{i}.affine, n, n),  ...
    problem.constraints{i}.variables-1);
  end
else
    n = problem.K.s(1);
    m = size(A, 1);
    p.AddDenseLinearMatrixInequality(reshape(A', n, n*m) , reshape(full(c), n, n));
end

p.options.inv_sqrt_mu_max = 1000;
p.options.infeasibility_threshold = 1e3;
p.options.max_iteration = 25;
p.options.prepare_dual_variables = 1;
p.options.divergence_upper_bound = 1;
p.options.prepare_dual_variables = 1;
p.options.final_centering_steps = 5;

tic;
[conex_primal, conex_dual, solved] = p.Maximize(problem.b);
info.cpusec = toc;

info.dinf = ~solved;
info.pinf = ~solved;
if pars.blkdiag
  [x, y] = problem.ConexPostProcess(conex_primal, conex_dual);
else
  y = conex_primal;
  x = conex_dual{1};
end
y = T * y;

if pars.errors
  info.errors(1) = abs(c'*x - bin'*y);
  info.errors(2) = c'*x - bin'*y;
end



function y = IsNontrivialField(K, f)
  if isfield(K, f) 
    if isempty(getfield(K, f))
      y = 0
    else
      y = getfield(K, f) > 0;
    end
  else
    y = 0;
  end
