function S_irls = irls_inp(A, rhs, DX, DY, ind_row,it_lsqr,it_irls, S0)
% INPUTS:
% A                = forward operator of the linear system
% rhs              = right hand side of the linear system
% DX,DY            = gradient matrices
% ind_row          = row indexes correspondent to the application of the gradients
% it_lsqr          = number of iterations of LSQR
% it_irls          = number of iterations of IRLS
% S0               = initial solution
% OUTPUT:
% S_irls           = result of l1 regularization by irls

S_irls = zeros(size(A,2),1);                  
if ~exist('S0','var')
    S0 = zeros(size(S_irls));    
end
%%% IRLS Loop
for i = 1:it_irls
    ww = repmat(((sqrt((DX*(S0(:)+S_irls)).^2+(DY*(S0(:)+S_irls)).^2))),2,1);
    ww = ww./(ww.^2+eps);
    W_vect = ones(length(A),1);
    if i ~= 1
        W_vect(ind_row) = ww;
    end
    W = spdiags(sqrt(W_vect),0,length(W_vect),length(W_vect));
    S_irls = lsqr(W*A,W*rhs,1e-20,it_lsqr,[],[],S_irls(:));
end
