function [A,b,T]= CleanLinear(A,b,useQR)
  
    if (size(A,1) ~= length(b))
       error('Number of rows of A and b do not match.') 
    end

    if ~exist('useQR','var')
        useQR = 0;
    end

    if (useQR)
        R = qr(sparse([A,b]'));
        [r,c] = find(R);
        %the first non-zero entry on a row implies
        %the column (e.g. equation) is linearly independent
        [~,indx] = unique(r,'first');
        eqKeep = c(indx);
    else
        eqKeep = find(any([A,b],2));
    end
     
    %a map that maps dual variables for original system to updated
    %Dual variable for equation we've removed gets mapped to 0
    %Dual variable for equation we keep gets mapped to itself
    numEqRed = length(eqKeep);
    numEqOrig = size(A,1);
    T = sparse(eqKeep,1:numEqRed,1,numEqOrig, numEqRed);

    A = A(eqKeep,:);
    b = b(eqKeep,:);
