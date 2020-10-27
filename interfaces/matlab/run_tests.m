function run_tests(path_to_conex_library)
  addpath(path_to_conex_library)
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
  [y, status] = p.Maximize(b);
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
  [y, status] = p.Maximize(b);
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
  [y, status] = p.Maximize(b);
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
  [y, status] = p.Maximize(b);
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

