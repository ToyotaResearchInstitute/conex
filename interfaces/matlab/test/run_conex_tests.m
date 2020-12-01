function run_tests(path_to_conex_library)
  if nargin > 0
    addpath(path_to_conex_library)
  end

  LPTests();
  SDPTests();
  SparseTests();


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

function e = dimacs(A, b, c, x, y) 
  e(1) = norm(A*x(:)-b)/(1+norm(b(:),'inf'));
  e(2) = (c(:)'*x - b'*y)/(1 + abs(c(:)'*x) + abs(b'*y));

