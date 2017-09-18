function img_inp = inpainting(img_orig, B, reg_alg)
% INPUTS:
% img_orig    = original image (2D array, double 0-1)
% B           = size of B x B region of the pixel selector
% reg_alg     = regularization algorithm
% OUTPUTS:
% I_inp       = inpainted image

%%% image size
[R, C, colors] = size(img_orig);

%% BUILD GRADIENTS & PIXEL SELECTORS

%%% gradient step size
delta = 3; 
%%% period of the pixel selector
step_size = max([ceil(((delta+1)*2)/B)*B, 2*B]);
%%% number of pixel selectors
n_ps = step_size^2/B^2;
%%% padding size (in order to have image size = k*B)
pad_size = [ceil(R/B)*B + step_size*2, ceil(C/B)*B + step_size*2];
%%% build pixel selector for inpainting
ps = build_ps([ceil(R/B)*B, ceil(C/B)*B], step_size, B, n_ps);
%%% build gradient for inpainting
[Dx, Dy] = build_grad(pad_size, delta);
D = [Dx; Dy];

%% REGULARIZATION PARAMETERS

%%% penalty weight
mu_g =  1e-20;
%%% regularizer matrix
reg = sqrt(mu_g)*D;
%%% regularizer right hand side
rhs_reg = sparse(size(reg, 1), 1);

%% INPAINTING

%%% padding of original image
img_orig_pad = padarray(img_orig, ...
    [ B*ceil(size(img_orig,1)/B)-size(img_orig,1), ...
    B*ceil(size(img_orig,2)/B)-size(img_orig,2) ], ...
    'symmetric','post');
img_orig_pad = padarray(img_orig_pad, [step_size, step_size], 'symmetric');
%%% set dynamics of images
max_orig = max(img_orig_pad(:));
min_orig = min(img_orig_pad(:));
%%% for saving intermediate values
img_temp_1 = zeros(size(img_orig_pad));
%%% loop over the different pixel selectors
for m = 1:n_ps
    %%% take the m-th pixel selector
    ps_m = ps(:,:,m);
    %%% padding, in order to have the same image size
    ps_m = padarray(ps_m, [step_size, step_size], 1);
    %%% masking of the image
    img_masked = bsxfun(@times, img_orig_pad, ps_m);
    %%% initialize temporal image, for intermediate results
    img_temp_2 = zeros(prod(pad_size), colors);
    %%% building the matrix for solving the linear system: data term
    H = speye(prod(pad_size));
    H((ps_m(:) == 0),:) = 0;
    %%% overall matrix
    A = [H; reg];
    %%% loop over the color planes
    for c = 1:colors
        rhs = [reshape(img_masked(:,:,c), [prod(pad_size), 1]); rhs_reg];
        %%% check regularization strategy
        if strcmp(reg_alg, 'TV')
            %%% TV solution
            it_lsqr = 100;
            it_irls = 4; 
            img_temp_2(:, c) = irls_inp(A, rhs, ...
                D(1:size(D, 2), :), D(size(D, 2)+1:end, :), ...
                size(H, 1)+1:size(A, 1), it_lsqr, it_irls);
        else
            %%% L2 solution
            it_lsqr = 1e3;
            tol_lsqr = 1e-20; 
            img_temp_2(:,c) = lsqr(A, rhs, tol_lsqr, ...
                it_lsqr, [], [], ...
                reshape(img_orig_pad(:, :, c), [prod(pad_size), 1]));
        end
    end
    %%% resizing of the image, and adjusting the dynamics
    img_temp_2 = reshape(img_temp_2, [pad_size, colors]);
    img_temp_2 = (img_temp_2 - min(img_temp_2(:)))./...
        (max(img_temp_2(:) - min(img_temp_2(:))));
    img_temp_2 = img_temp_2.*(max_orig - min_orig) + min_orig;
    %%% selecting only inpainted pixels, corresponding to the ps_m
    img_temp_1 = bsxfun(@times, img_temp_1, ps_m) + ...
        bsxfun(@times, img_temp_2, ~ps_m);
end
%%% selecting only the R x C area of pixels
img_inp = img_temp_1(step_size+1:step_size+R, step_size+1:step_size+C, :);

end


%% PIXEL SELECTORS

function PS = build_ps(img_size, step_size, B, n_pss)
% INPUTS:
% img_size    = [Rows, Columns] of the image
% step_size   = period of the pixel selector
% B           = size of B x B region of the pixel selector
% n_pss       = number of pixel selectors
% OUTPUTS:
% PS          = pixel selector matrix


PS = false(img_size(1), img_size(2), n_pss);
f = zeros(2*B);
f(1:B,1:B) = 1;
cnt = 1;
for r = 1:sqrt(n_pss)
    for c = 1:sqrt(n_pss)
        %%% build the basic structure
        ps = false(img_size);
        ps((r-1)*B+1:step_size:end,(c-1)*B+1:step_size:end) = true;
        %%% create the mask
        ps = imfilter(ps,f);
        %%% take the logical complement
        PS(:, :, cnt) = ~ps;
        cnt = cnt +1;
    end
end
end


%% GRADIENTS

function [Dx, Dy] = build_grad(img_size, delta)
% INPUTS:
% img_size    = [Rows, Columns] of the image
% delta       = gradient step size
% OUTPUTS:
% Dx          = horizontal gradient 
% Dy          = vertical gradient

R = img_size(1);
C = img_size(2);
%%% horizontal derivative vector
ones_x = [-ones(R*C, 1),  ones(R*C, 1)];
%%% computing Dx
Dx = spdiags(ones_x, [-delta*R, 0], R*C, R*C)';
Dx(R*C - delta*R+1:end, :) = 0;
%%% computing Dy
mat_y = spdiags(ones_x, [-delta, 0], R, R);
Dy = kron(speye(C), mat_y)';
for i = 0:delta-1
    Dy(R-i:R:end, :) = 0;
end

end
