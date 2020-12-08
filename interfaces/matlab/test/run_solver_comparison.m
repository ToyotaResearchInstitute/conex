function run_solver_comparison
  n = [20, 50, 100]
  num_problems = 1;

  % Comparison: increase m linearly with n.
  m = n;
  for i = 1:length(n)
    ni = n(i);
    mi = m(i);
    [errP(i, :), errG(i, :), time(i, :), eigX(i, :), eigS(i, :)] = SolverComparison(ni, mi, num_problems);
  end

  for i = 1:length(n)
    ni = n(i);
    mi = m(i);
    errPi = errP(i, :);
    errGi = errG(i, :);
    timei = time(i, :);
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
    [errP(i, :), errG(i, :), time(i, :), eigX(i, :), eigS(i, :)] = SolverComparison(ni, mi, num_problems);
  end

  for i = 1:length(n)
     ni = n(i);
    mi = m(i);
    errPi = errP(i, :);
    errGi = errG(i, :);
    timei = time(i, :);
    eigXi = eigX(i, :);
    eigSi = eigX(i, :);
    fprintf('(%d, %d) & %1.1d & %1.1d &  %1.1d & %1.1d & %1.1d & %1.1d & %1.1d & %1.1d & %1.1d & %1.1d \\\\ \n', ni, mi, timei, errPi, errGi, eigXi, eigSi)
  end

function [errPrimal_cs, errGap_cs, time_cs, errEigX, errEigS] = SolverComparison(n, m, num)
  K.s = n;
  sym = @(x) x + x';
  vect = @(x) x(:);

  %The 0-pass is ignored since its timings are outliers (likely Matlab JIT related).
  for t=0:num
    A = zeros(m, n*n);
    for i = 1:m
      A(i, :) = vect(sym(randn(n, n)))';
    end

    x = eye(n, n);
    b = A * x(:);
    c = vect(eye(n, n));

    [xc, yc, info] = conex(A, b, c, K);

    %Ignore first pass
    if (t > 0)
      % Get time after conversion of SeDuMI format.
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
      [blk, Avec, C, b, perm] = read_sedumi(A, b, c, K);
      options = sqlparameters;
      options.vers = 2;
      tic;
      [~, xs, ys] = sdpt3(blk, Avec, C, b, options);
      xs = xs{1}(:);
      ys = ys(:);
      % Timing for first-pass is an outlier (likely Matlab JIT related).
      if (t > 0)
        time_sdpt3(t) = toc;
        errPs(t) = norm(A*xs-b)/(1+norm(b(:), 'inf'));
        errPc(t) = norm(A*xc-b)/(1+norm(b(:), 'inf'));

        errGaps(t) =   (c(:)'*xs - b'*ys)/(1 + abs(c(:)'*xs) + abs(b'*ys));
        errGapc(t) =   (c(:)'*xc - b'*yc)/(1 + abs(c(:)'*xc) + abs(b'*yc));

        eigXs(t) = min(eigK(xs, K))
        eigXc(t) = min(eigK(xc, K));
        eigSs(t) = min(eigK(c-A'*ys, K));
        eigSc(t) = min(eigK(c-A'*yc, K));
      end
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

