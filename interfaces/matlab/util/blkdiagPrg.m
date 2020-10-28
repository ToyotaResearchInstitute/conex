classdef blkdiagPrg

properties
  A
  b
  c
  K
  Ty
  indx
  clique
  dim
  M
  unreducedPrg;
end

methods
  function self = blkdiagPrg(A, b, c, K)
    cone = coneBase(K);
    K = cone.K;
    A = cone.Symmetrize(A); 
    c = cone.Symmetrize(c(:)'); 
    unreducedPrg.A = A;
    unreducedPrg.b = b;
    unreducedPrg.c = c;
    unreducedPrg.K = K;
    [clique, Ar, self.c, self.K, indx, M] = BuildMask(unreducedPrg.A, unreducedPrg.b, ...
                                                      unreducedPrg.c, unreducedPrg.K);

    [self.A, self.b, T] = CleanLinear(Ar, unreducedPrg.b);
    self.unreducedPrg = unreducedPrg;
    self.clique = clique;

    self.dim(1) = length(indx);
    self.dim(2) = size(unreducedPrg.A, 2);
    self.indx = indx;
    self.M = M;
    self.Ty = T;
  end

  function [xr, yr] = Recover(self, x, y)
    xr = self.RecoverPrimal(x);
    yr = self.RecoverDual(y);
  end

  function [xr] = RecoverPrimal(self, x)
    cone = coneBase(self.unreducedPrg.K);
    xr = full(sparse(ones(1, length(self.indx)), self.indx, x, 1, cone.NumVar)');
  end

  function [yr] = RecoverDual(self, yinput, eps)
    yr = self.Ty * yinput;
  end
end % methods
end
