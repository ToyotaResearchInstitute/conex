function run_tests(path_to_conex_library)
  if nargin > 0
    addpath(path_to_conex_library)
  end

  %LPTests();
  %SDPTests();
  %SparseTests();
  n = 10;
  %m = round(.1*n*n);
  m = 10%round(n);
  num_problems = 1; 
  SedumiTests(n, m, num_problems);

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



function SedumiTests(n, m, num)


  K.s = n;
  sym = @(x) x + x';
  vect = @(x) x(:);
  for t=1:num
    A = zeros(m, n*n);
    for i = 1:m
      A(i, :) = vect(sym(randn(n, n)))';
    end
    %x = randn(n, n); x = x*x';
    %x = eye(n, n); x = x*x';
    x = eye(n, n); 
    b = A * x(:);
    c = vect(eye(n, n));

    [xc, yc, info] = conex(A, b, c, K);
    time_conex(t) = info.cpusec;

    if 0
    time_sedumi = -1;
    if exist('sedumi', 'file') == 2 %a matlab file name sedumi exists
      tic
      [xs, ys] = sedumi(A, b, c, K);
      time_sedumi(t) = toc;
      errPs(t) = norm(A*xs-b);
      errPc(t) = norm(A*xc-b);
      eigXs(t) = min(eigK(xs, K))
      eigXc(t) = min(eigK(xc, K));
      eigSs(t) = min(eigK(c-A'*ys, K));
      eigSc(t) = min(eigK(c-A'*yc, K));
    end
    end

    if exist('sdpt3', 'file') == 2 %a matlab file name sedumi exists
      [blk,Avec,C,b,perm] = read_sedumi(A,b,c,K);
      options = sqlparameters;
      options.vers = 2;
      profile on
      tic;
      [~, xs, ys] = sdpt3(blk, Avec, C, b, options);
      xs = xs{1}(:);
      ys = ys(:);
      time_sdpt3(t) = toc;
      profile off
      errPs(t) = norm(A*xs-b)/(1+norm(b(:),'inf'));
      errPc(t) = norm(A*xc-b)/(1+norm(b(:),'inf'));

      errGaps(t) =   (c(:)'*xs - b'*ys)/(1 + abs(c(:)'*xs) + abs(b'*ys));
      errGapc(t) =   (c(:)'*xc - b'*yc)/(1 + abs(c(:)'*xc) + abs(b'*yc));

      eigXs(t) = min(eigK(xs, K))
      eigXc(t) = min(eigK(xc, K));
      eigSs(t) = min(eigK(c-A'*ys, K));
      eigSc(t) = min(eigK(c-A'*yc, K));
      fprintf('|e_p, e_g| Conex %d\n', dimacs(A,b,c,xc,yc));
      fprintf('|e_p, e_g| SDPT3 %d\n', dimacs(A,b,c,xs,ys));
    end
  end

      fprintf('Times: SDPT3 %d,  Conex %d \n', [mean(time_sdpt3), mean(time_conex)]);
      fprintf('|Ax-b|: SDPT3 %d,  Conex %d \n', [mean(errPs), mean(errPc) ]);
      fprintf('Gap |x|: SDPT3 %d,  Conex %d \n', [mean(errGaps), mean(errGapc)]);

      fprintf('eig min |x|: SDPT3 %d,  Conex %d \n', [mean(eigXs), mean(eigXc)]);
      fprintf('eig min |s|: SDPT3 %d,  Conex %d \n', [mean(eigSs), mean(eigSc)]);

      fprintf('Times: SDPT3 %d,  Conex %d \n', [var(time_sdpt3), var(time_conex)]);
      fprintf('|Ax-b|: SDPT3 %d,  Conex %d \n', [var(errPs), var(errPc) ]);
      fprintf('Gap: SDPT3 %d,  Conex %d \n', [var(errGaps), var(errGapc)]);

      fprintf('eig min |x|: SDPT3 %d,  Conex %d \n', [var(eigXs), var(eigXc)]);
      fprintf('eig min |s|: SDPT3 %d,  Conex %d \n', [var(eigSs), var(eigSc)]);
