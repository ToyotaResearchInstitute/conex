classdef ConexPreprocess

  properties
    TransformDual = [];
    OffsetDual = [];
    TransformPrimal = [];
    Ao = [];
    bo = [];
    Ko = [];
    num_free = [];
    constraints;
    b;
    K;
    r;
  end

  methods
    function self = ConexPreprocess(A, b, c, K)
      self.Ao = A;
      self.bo = b;
      self.Ko = K;
      self.num_free = 0;
      if isfield(K, 'f')
        self.num_free = K.f;
        [A, b, c, K, self.TransformDual, self.OffsetDual] = EliminateFreeVars(A, b, c, K);
      end
      self.r = blkdiagPrg(A, b, c, K);

      self.b = self.r.b;
      self.K = self.r.K;
      self.constraints = ExtractConstraintMatrices(self.r.A, self.r.c, self.r.K, 1);
    end

    function [x, y] = ConexPostProcess(self, conex_primal, conex_dual)
        xcell = conex_dual;
        y = conex_primal;
        K = self.r.K;
        num_free = self.num_free;

        s = 1;
        x = zeros(size(self.r.A,2), 1);
        for i = 1:length(K.s)
          n = K.s(i);
          e = s + K.s(i)  * K.s(i) ;
          x(s:e -1) = xcell{i}(:);
          s = e;
        end

        [x, y] = self.r.Recover(x, y);
        if num_free > 0
          xf = self.Ao(:, 1:num_free)\(self.bo-self.Ao(:, num_free + 1 :end) * x);
          x = [xf; x];
          y = self.OffsetDual + self.TransformDual * y;
        end
    end
  end

end
