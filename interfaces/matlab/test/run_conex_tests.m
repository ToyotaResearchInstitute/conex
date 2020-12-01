function run_tests(path_to_conex_library)
  if nargin > 0
    addpath(path_to_conex_library)
  end

  %LPTests();
  %SDPTests();
  %SparseTests();
  SedumiTests();

function LPTests()
  p = ConexProgram();

  num_var = 2;
  num_constraint = 4;

  A = rand(num_constraint, num_var);
  c = ones(num_constraint, 1); 
  b = A' * ones(num_constraint, 1);

  p.AddLinearInequality(A, c);
  [y, x, status] = p.Maximize(b);
  return 
  if ~status
    error('Test failed: expected feasible solution found')
  end

  % Do infeasibility test.
  % 1 >= y1,  1 >= -y1
  p = ConexProgram();
  A = [1, 0; ...
       0, 1; ...
      -1, 0; ...
      0, -1];
  c = [1, 1, -2, 1]';

  p.AddLinearInequality(A, c);
  [y, x, status] = p.Maximize(b);
  if status
    error('Test failed: expected infeasibility')
  end

function SDPTests()
  p = ConexProgram();

  num_var = 1;

  n = 3;
  A = [1, 0, 0,  1, 1, 0;
       0, 1, 0,  1, 1, 0;
       0, 0, 1,  0, 0, 0]; 

  c = eye(n);

  b = [1, 1]; 

  p.AddLinearMatrixInequality(A, c);
  [y, x, status] = p.Maximize(b);
  if ~status
    error('Test failed: expected feasible solution found')
  end

  slack = c - A * [y(1) * eye(n); y(2) * eye(n)];
  if min(eig(slack)) < 0
      error('Test failed: slack not psd.')
  end


function SparseTests()
  p = ConexProgram();

  num_var = 1;

  n = 3;
  A = [1, 0, 0,  1, 1, 0;
       0, 1, 0,  1, 1, 0;
       0, 0, 1,  0, 0, 0]; 

  c = eye(n);

  b = [1, 1, 1]; 

  p.AddLinearMatrixInequality(A, c, [0, 1]);
  p.AddLinearMatrixInequality(A, c, [1, 2]);
  [y, x, status] = p.Maximize(b);
  if ~status
    error('Test failed: expected feasible solution found')
  end

  slack1 = c - A * [y(1) * eye(n); y(2) * eye(n)];
  slack2 = c - A * [y(2) * eye(n); y(3) * eye(n)];
  if min(eig(slack1)) < 0
      error('Test failed: slack not psd.')
  end
  if min(eig(slack2)) < 0
      error('Test failed: slack not psd.')
  end

  b1 = zeros(3, 1);
  b2 = zeros(3, 1);
  b1(1:2) = reshape(A, n*n, 2)'  * x{1};
  b2(2:3) = reshape(A, n*n, 2)'  * x{2};
  if (norm(b1 + b2 - b(:)) > 1e-12)
      error('Test failed: dual constraints violated.')
  end
  
function SedumiTests()
  m = 15;
  n = 150;
  K.s = n;
  sym = @(x) x + x';
  vect = @(x) x(:);
  for i = 1:m
    A(i, :) = vect(sym(randn(n, n)))';
  end
  %x = randn(n, n); x = x*x';
  %x = eye(n, n); x = x*x';
  x = eye(n, n); 
  b = A * x(:);
  c = vect(eye(n, n));

  [xc, yc, info] = conex(A, b, c, K);
  time_conex = info.cpusec;

  time_sedumi = -1;
  if exist('sedumi', 'file') == 2 %a matlab file name sedumi exists
    tic
    [xs, ys] = sedumi(A, b, c, K);
    time_sedumi = toc;
    fprintf('Times: Sedumi %d,  Conex %d \n', [time_sedumi, time_conex]);
    fprintf('|Ax-b|: Sedumi %d,  Conex %d \n', [norm(A*xs-b), norm(A*xc-b)]);
    fprintf('eig min |x|: Sedumi %d,  Conex %d \n', [min(eigK(xs, K)), min(eigK(xc, K))]);
    fprintf('eig min |s|: Sedumi %d,  Conex %d \n', [min(eigK(c-A'*ys, K)), min(eigK(c-A'*yc, K))]);
  end

  time_sdpt3 = -1;
  if exist('sdpt3', 'file') == 2 %a matlab file name sedumi exists
    [blk,Avec,C,b,perm] = read_sedumi(A,b,c,K);
    options = sqlparameters;
    options.vers = 2;
    tic;
    [~, xs, ys] = sdpt3(blk, Avec, C, b, options);
    xs = xs{1}(:);
    ys = ys(:);
    time_sdpt3 = toc;
    fprintf('Times: SDPT3 %d,  Conex %d \n', [time_sdpt3, time_conex]);
    fprintf('|Ax-b|: SDPT3 %d,  Conex %d \n', [norm(A*xs-b), norm(A*xc-b)]);
    fprintf('eig min |x|: SDPT3 %d,  Conex %d \n', [min(eigK(xs, K)), min(eigK(xc, K))]);
    fprintf('eig min |s|: SDPT3 %d,  Conex %d \n', [min(eigK(c-A'*ys, K)), min(eigK(c-A'*yc, K))]);
  end

