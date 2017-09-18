%% IMAGE ANONYMIZATION PIPELINE

% Reference: S. Mandelli, L. bondi, S. Lameri, V. Lipari, P. Bestagini, S. Tubaro,
% "Inpainting-Based Camera Anonymization"
% IEEE International Conference on Image Processing (ICIP), 2017. 

% The code shows an example of image anonymization pipeline.
% Author: Sara Mandelli

close all;
clearvars;
clc; 

%% ADDPATH

addpath(genpath('CameraFingerprint')); 
addpath('functions'); 
addpath(genpath('BM3D')); 

%% LOAD IMAGE

img_orig = im2double(imread('img_orig.png'));

%% CHOOSE SIZE OF B X B REGION OF THE PIXEL SELECTOR [3, 5]

B = 5; 3; 

%% CHOOSE THE REGULARIZATION STRATEGY ['L2', 'TV']

reg_alg = 'L2'; 'TV';  

%% IMAGE INPAINTING

img_inp = inpainting(img_orig, B, reg_alg); 
imwrite(img_inp, 'img_inp.png'); 

%% EDGE RECONSTRUCTION

img_anonymized = edge_reconstruction(img_orig, img_inp);
imwrite(img_anonymized, 'img_anonymized.png'); 

%% EVALUATE PSNR

PSNR_anonymized = psnr(img_anonymized, img_orig);

%% EVALUATE CORRELATION WITH CAMERA PRNU

%%% load camera PRNU
load('camera_prnu.mat'); 
%%% read images as double, range in 0-255
img_orig_255 = double(imread('img_orig.png')); 
img_anonymized_255 = double(imread('img_anonymized.png')); 
%%% compute correlations
ncc_orig = ncc_prnu(img_orig_255, prnu); 
ncc_anonymized = ncc_prnu(img_anonymized_255, prnu); 

%% PERFORMANCE EVALUATION

%%% load pre-computed true positive correlations
load('NCC_tp.mat', 'ncc_tp'); 
%%% load pre-computed true negative correlations
load('NCC_tn.mat', 'ncc_tn'); 

%% PLOT RESULTS

%%% original image
figure(1); 
imshow(img_orig);
title('Original image', 'fontsize', 16);

%%% anonymized image
figure(2);
imshow(img_anonymized); 
title('Anonymized image', 'fontsize', 16);

%%% plot histograms of correlations
figure(3);  
colOrd = get(gca, 'ColorOrder');
histogram(ncc_tp, 100, 'faceAlpha', 0.5, 'edgeAlpha', 0.5, 'edgecolor', ...
    colOrd(1, :));
hold on;
histogram(ncc_tn, 100, 'faceAlpha', 0.5, 'edgeAlpha', 0.5,'edgecolor',...
    colOrd(2, :));
hold on;
plot([ncc_orig, ncc_orig], [0, 25],'color',[1 0 1],'linewidth', 3);
hold on;
plot([ncc_anonymized, ncc_anonymized], [0, 25],'color', [0 1 0], 'linewidth', 3);
ll = legend({'True Positive Set','True Negative Set','Original \rho', ... 
    'Anonymized \rho'},'fontsize',14);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',14)
xlabel('\rho ( W_{I}, I \cdot PRNU)','fontsize',16);
set(gca,'yticklabel',{[]}); 
title(sprintf('Anonymized PSNR: \n %2.2f dB', PSNR_anonymized), 'fontsize', 16);
pbaspect([15 5 5]); 

%% FUNCTIONS

function I_edge = edge_reconstruction(img_1, img_2, edge_type, seed)
% INPUTS:
% img_1       = edge-providing image (2D array, double 0-1)
% img_2       = image to be modified in its edges (2D array, double 0-1)
% edge_type   = edge detector to use (string of chars)
% seed        = seed for random variable generation
% OUTPUTS:
% I_edge      = edge-modified image: we put the edges of img_1 into img_2
% for the code of the function CBM3D, please download it at the link
% http://www.cs.tut.fi/~foi/GCF-BM3D/#ref_software

%% DEFAULT FUNCTION PARAMETERS

sigma = 7; 
profile = 'lc'; 
strel_type = 'disk'; 
strel_size = 3; 
if nargin < 3
    edge_type = 'canny';
end
if nargin < 4
    seed = 'default';
end
%%% set the seed
rng(seed);

%% DENOISING OF img_1

%%% first step of BM3D
[~, I_den] = CBM3D(1, img_1, sigma, profile);
%%% second step of BM3D
[~, I_den] = CBM3D(1, I_den, sigma, profile);

%%% if you cannot download CBM3D, you can directly load the result
% I_den = im2double(imread('img_orig_den.png')); 

%% EDGE RECONSTRUCTION 

%%% extract edges from I_dem
Edges = edge(rgb2gray(I_den), edge_type) ;
%%% imdilate with a disk structure element
Edges = imdilate(Edges, strel(strel_type, strel_size));
N_edges = ~Edges;
%%% sum the 2 image contributions
I_edge = bsxfun(@times, img_2, N_edges)+ bsxfun(@times, I_den, Edges);

end

function [ncc_img] = ncc_prnu(img, prnu)
% INPUTS:
% img         = input image (double 0-255)
% prnu        = camera prnu
% OUTPUTS:
% ncc_img     = normalized crosscorrelation with camera prnu

%%% noise of original image
k_img = noise_extraction(img);
%%% element-wise product with camera prnu
w_img = rgb2gray1(img).*prnu;
%%% crosscorrelation
ncc_img = cross_corr_prnu(w_img(:), k_img(:));

end

function k_img = noise_extraction(img)
% INPUTS:
% img         = input image (double 0-255)
% OUTPUTS:
% k_img       = extracted noise
% for the code of the functions, please download the package at the link
% http://dde.binghamton.edu/download/camera_fingerprint/

%%% RGB images
W = zeros(size(img));
qmf = MakeONFilter('Daubechies',8);
%%% local std of extracted noise
sigma = 3; 
%%% number of decomposition levels
L = 4; 
for c = 1:size(img, 3)
    W(:,:,c) = NoiseExtract(img(:,:,c), qmf, sigma, L);
end
k_img = ZeroMeanTotal(W);
k_img = single(k_img);
k_img = rgb2gray1(k_img);
k_img = WienerInDFT(k_img, std2(k_img));
k_img = single(k_img);

end

function ncc = cross_corr_prnu(prnu_1, prnu_2)
% INPUTS:
% prnu_1      = first prnu (1D array)
% prnu_2      = second prnu (1D array)
% OUTPUTS:
% ncc         = crosscorrelation between prnu_1 and prnu_2

%%% convert to single precision
prnu_1 = single(prnu_1);
prnu_2 = single(prnu_2);
%%% L2 norms
prnu_1_norm = norm(prnu_1); 
prnu_2_norm = norm(prnu_2); 
%%% correlate
ncc = prnu_1'*prnu_2;
%%% normalize
ncc = ncc./(prnu_1_norm*prnu_2_norm+eps);

end


