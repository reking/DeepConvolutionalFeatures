function [scores, maxlabel] = feature_mat(im, use_gpu)
% scores = matcaffe_demo(im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format:
%   % convert from uint8 to single
%   im = single(im);
%   % reshape to a fixed size (e.g., 227x227)
%   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % permute from RGB to BGR and subtract the data mean (already in BGR)
%   im = im(:,:,[3 2 1]) - data_mean;
%   % flip width and height to make width the fastest dimension
%   im = permute(im, [2 1 3]);

% If you have multiple images, cat them with cat(4, ...)

% The actual forward function. It takes in a cell array of 4-D arrays as
% input and outputs a cell array. 
model_def_file = 'hanet_lenet.prototxt';
model_file = 'hanet5_iter_30000.caffemodel';


% init caffe network (spews logging info)
%if exist('use_gpu', 'var')
  %matcaffe_init(use_gpu);
  
 matcaffe_init(0, model_def_file, model_file);
%else
%  matcaffe_init();
%end
%matcaffe_init(use_gpu, model_def_file, model_file);
fprintf('ok');
%caffe('init', model_def_file, model_file);

% prepare oversampled input
% input_data is Height x Width x Channel x Num
tic;
input_data = {prepare_image(im)};
%save('input_data')
toc;
% do forward pass to get scores
% scores are now Width x Height x Channels x Num
tic;
scores = caffe('forward', input_data);
toc;
%save('score1.mat','scores');
scores = scores{1}
%size(scores)
scores = squeeze(scores);
scores = mean(scores,2);
save('scores');

[~,maxlabel] = max(scores);

% ------------------------------------------------------------------------
function images = prepare_image(im)
% ------------------------------------------------------------------------
%d = load('ilsvrc_2012_mean');
%IMAGE_MEAN = d.image_mean;
IMAGE_DIM = 30;
%CROPPED_DIM = 227;

% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
%im = im(:,:,[3 2 1]) - IMAGE_MEAN;

% oversample (4 corners, center, and their x-axis flips)
images = zeros(IMAGE_DIM, IMAGE_DIM, 1, 1, 'single');
%for i =1:32
    images(:,:,:,1)= im;
%end



