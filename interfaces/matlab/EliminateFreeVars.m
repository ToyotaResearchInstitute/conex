function [A,b,c,K, B, y0] = EliminateFreeVars(A, b, c, K)

  s = 1;
  e = s + K.f - 1;

  Af = A(:, 1:e)';
  cf = c(1:e);

  A = A(:, e+1:end);
  c = c(e+1:end);

  y0 = Af\cf;
  B = spnull(Af);
  c = c(:) - [A'*y0];
  A = [A'*B]';
  b = [b'*B]';

  K.f = 0;
