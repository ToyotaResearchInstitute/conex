function [cliques,Ar,cr,Kr,indx,M] = BuildMask(A,b,c,K)

if K.q + K.r  > 0
    error('Lorentz cone constraints not yet supported.');
end

M = full(c~=0);
M = M(:);
nnz_M = nnz(M);
cone = coneBase(K);
Kr = K;

cliques = {};

while(1)

    [M] = SubspaceClosureCoordDisjointSupport(M,A,b);  %conservative, but fast
   %[M,subspace] = SubspaceClosureCoord(M,subspace,K); %requires computing large inverse

    for i=1:length(K.s)
        [s,e] = cone.GetIndx('s',i);
        [temp,cliques_i] = BinaryPsdCompletion(reshape(M(s:e), K.s(i), K.s(i)));
        M(s:e) = temp(:);
        Kr.s(i) = length(cliques_i);
        cliques{i} = cliques_i;
    end
    
    if (nnz_M == nnz(M))
       break; 
    end
        
    nnz_M = nnz(M);
    
end


[s,e] = cone.GetIndx('f',1);
indx = find(M(s:e));
Kr.f = length(indx);

[s,e] = cone.GetIndx('l',1);
indx = find(M(s:e));
Kr.l = length(indx);

[s,e] = cone.GetIndx('s',1);
indx =  find(M(1:s-1));

Kr.s = [];
for i=1:length(K.s)
    for j=1:length(cliques{i})
        indx = [indx;cone.SubMatToIndx(cliques{i}{j},i  )];
        Kr.s(end+1) = length(cliques{i}{j});
    end
end

Ar = A(:,indx);
br = b;
cr = c(indx);

end


function [M] = SubspaceClosureCoordDisjointSupport(M,A,b)

    isSparse = issparse(M);
    M = M';
    M = any(A(b~=0,:),1) | M; %the stuff we must pass through
    nnz_M = nnz(M);
    not_converged = 1;
    while not_converged
        tau = any(A(:,M>0)~=0,2); %all rows at least partially passed through
        M = any(A(tau,:),1)';
        if nnz_M ~= nnz(M)
            nnz_M = nnz(M);
        else
            not_converged = 0;
        end
    end
    %M will be sparse if A is sparse.  Undo to match input.
    if ~isSparse
        M = full(M); 
    end
end


