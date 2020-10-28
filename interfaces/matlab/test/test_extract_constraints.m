function pass = test_extract_constraints()
m = 2;
K.s = [2;2];
n = sum(K.s.*K.s);

A = [1, 2, 2, 1, 0, 0, 0, 0; 
     0, 0, 0, 0, 2, 1, 1, 2;
     1, 3, 3, 1, 2, -3, -3, 2];

A(A<.5) = 0;
A = sparse(A);
c = randn(n, 1);
DoTest(A, c, K);

function pass = DoTest(A, c, K)
constraints = ExtractConstraintMatrices(A, c, K, 1);
A = full(A);
c = full(c);
s = 1;
for i = 1:length(K.s)
  n = K.s(i);
  e = s + K.s(i)  * K.s(i) ;
  Ai = A(:, s:e-1)';
  [~, vars, ~] = find(Ai);
  vars = sort(unique(vars));
  Ai = Ai(:, vars);
  Ai = reshape(Ai, n, length(vars)*n);
  ci = reshape(c(s:e-1), n, n);

  Ai2 = constraints{i}.matrix_conex_format;
  if (norm(Ai2 - Ai) > 1e-12) || norm(constraints{i}.variables - vars) > 1e-12 || ...
     norm(ci(:) - constraints{i}.affine(:)) > 1e-12
      error('Test failed');
  end
  s = e;
end

