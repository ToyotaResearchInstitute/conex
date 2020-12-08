function run_tests(path_to_conex_library)
  if nargin > 0
    addpath(path_to_conex_library)
  end

  %LPTests();
  %SDPTests();
  %SparseTests();

  n = [20, 50, 100]
  num_problems = 1; 

  % Comparison: increase m linearly with n.
  m = n;
  for i = 1:length(n)
    ni = n(i);
    mi = m(i);
    [errP(i, :), errG(i, :), time(i, :), eigX(i,:), eigS(i, :)] = SolverComparison(ni, mi, num_problems);
  end

  for i = 1:length(n)
    ni = n(i);
    mi = m(i);
    errPi = errP(i,:);
    errGi = errG(i,:);
    timei = time(i,:);
    eigXi = eigX(i, :);
    eigSi = eigX(i, :);
    fprintf('(%d, %d) & %1.1d & %1.1d &  %1.1d & %1.1d & %1.1d & %1.1d & %1.1d & %1.1d & %1.1d & %1.1d \\\\ \n', ni, mi, timei, errPi, errGi, eigXi, eigSi)
  end

  % Comparison: increase m quadratically. 
  m = round(.1*n.*n);
  for i = 1:length(n)
    ni = n(i);
    mi = m(i);
    num_problems = 1; 
    [errP(i, :), errG(i, :), time(i, :), eigX(i,:), eigS(i, :)] = SolverComparison(ni, mi, num_problems);
  end

  for i = 1:length(n)
    ni = n(i);
    mi = m(i);
    errPi = errP(i,:);
    errGi = errG(i,:);
    timei = time(i,:);
    eigXi = eigX(i, :);
    eigSi = eigX(i, :);
    fprintf('(%d, %d) & %1.1d & %1.1d &  %1.1d & %1.1d & %1.1d & %1.1d & %1.1d & %1.1d & %1.1d & %1.1d \\\\ \n', ni, mi, timei, errPi, errGi, eigXi, eigSi)
  end


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



function [errPrimal_cs, errGap_cs, time_cs, errEigX, errEigS] = SolverComparison(n, m, num)

  K.s = n;
  sym = @(x) x + x';
  vect = @(x) x(:);
  for t=0:num
    A = zeros(m, n*n);
    for i = 1:m
      A(i, :) = vect(sym(randn(n, n)))';
    end

    x = eye(n, n); 
    b = A * x(:);
    c = vect(eye(n, n));

    [xc, yc, info] = conex(A, b, c, K);
    if (t > 0)
    time_conex(t) = info.cpusec;
    end

    if 0
    time_sedumi = -1;
    if exist('sedumi', 'file') == 2 %a matlab .m exists
      tic
      [xs, ys] = sedumi(A, b, c, K);
      time_sedumi(t) = toc;
      if (t > 0)
      errPs(t) = norm(A*xs-b);
      errPc(t) = norm(A*xc-b);
      eigXs(t) = min(eigK(xs, K));
      eigXc(t) = min(eigK(xc, K));
      eigSs(t) = min(eigK(c-A'*ys, K));
      eigSc(t) = min(eigK(c-A'*yc, K));
      end
    end
    end

    if exist('sdpt3', 'file') == 2 %a matlab .m exists
      [blk,Avec,C,b,perm] = read_sedumi(A,b,c,K);
      options = sqlparameters;
      options.vers = 2;
      tic;
      [~, xs, ys] = sdpt3(blk, Avec, C, b, options);
      xs = xs{1}(:);
      ys = ys(:);
      % Timing for first-pass is an outlier (likely Matlab JIT related).
      if (t > 0)
        time_sdpt3(t) = toc;
        errPs(t) = norm(A*xs-b)/(1+norm(b(:),'inf'));
        errPc(t) = norm(A*xc-b)/(1+norm(b(:),'inf'));

        errGaps(t) =   (c(:)'*xs - b'*ys)/(1 + abs(c(:)'*xs) + abs(b'*ys));
        errGapc(t) =   (c(:)'*xc - b'*yc)/(1 + abs(c(:)'*xc) + abs(b'*yc));

        eigXs(t) = min(eigK(xs, K))
        eigXc(t) = min(eigK(xc, K));
        eigSs(t) = min(eigK(c-A'*ys, K));
        eigSc(t) = min(eigK(c-A'*yc, K));
      end
      fprintf('|e_p, e_g| Conex %d\n', dimacs(A,b,c,xc,yc));
      fprintf('|e_p, e_g| SDPT3 %d\n', dimacs(A,b,c,xs,ys));
    end
  end

      time_cs = [mean(time_sdpt3), mean(time_conex)];
      errPrimal_cs = [mean(errPs), mean(errPc) ];
      errDual_cs = [mean(errPs), mean(errPc) ];
      errGap_cs = [mean(errGaps), mean(errGapc)];

      errGap_cs = [mean(errGaps), mean(errGapc)];
      errEigX = [mean(abs(eigXs)), mean(abs(eigXc))];
      errEigS = [mean(abs(eigSs)), mean(abs(eigSc))];
  
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

