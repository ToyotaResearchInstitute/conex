%Calls conex using SeDuMI formatted inputs and outputs.
function [x, y, info] = conex(A, b, c, K, pars)

if isfield('l', K) && K.l > 0 
  error('Cone not supported yet');
end
if isfield('r', K) && K.r > 0 
  error('Cone not supported yet');
end
if isfield('q', K) && K.q > 0 
  error('Cone not supported yet');
end

if nargin < 5
  pars =[];
  pars.errors = 0;
end

info.numerr = 0;
info.pinf = 0;
info.dinf = 0;
info.feasratio = 1;
info.timing = [0, 0, 0]; 
info.cpusec = 0;

problem = ConexPreprocess(A, b, c, K);

p = ConexProgram();
for i = 1:length(problem.K.s)
  n = problem.K.s(i);
  Ai2 = problem.constraints{i}.matrix_conex_format;
  p.AddSparseLinearMatrixInequality(Ai2, reshape(problem.constraints{i}.affine, n, n),  ...
  problem.constraints{i}.variables-1);
  %p.AddDenseLinearMatrixInequality(Ai2, reshape(problem.constraints{i}.affine, n, n));
end

p.options.inv_sqrt_mu_max = 90000;
p.options.infeasibility_threshold = 1e10;
p.options.max_iteration = 25;
p.options.divergence_upper_bound = 2000;
p.options.prepare_dual_variables = 1;
p.options.final_centering_steps = 1;

tic;
[conex_primal, conex_dual, solved] = p.Maximize(problem.b);
info.cpusec = toc;
info.dinf = ~solved;
info.pinf = ~solved;
[x, y] = problem.ConexPostProcess(conex_primal, conex_dual);

if pars.errors
  info.errors(1) = abs(c'*x - b'*y);
  info.errors(2) = c'*x - b'*y;
end
