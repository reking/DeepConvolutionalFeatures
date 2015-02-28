%----------Extract the features from image database through caffe interface -----%
%featuremap = load('feature_new.mat');
files = dir('../../data/inkml/test_total/*.jpg');
%imagestring = ['../../data/inkml/test_feature/' files(1).name];
use_gpu = false;
%feature_new = feature_new.feature_new;
lenet_test_feature = zeros(66,41);
i=0;
labels = zeros(66,1);
for i = 0:66
imagestring = ['../../data/inkml/test_total/' files(i+1).name];
im = imread(imagestring);
size(im)
[score ,maxlabel] = feature_mat(im, use_gpu);
labels(i+1) = maxlabel;
lenet_test_feature(i+1,:) = score';
end
image_train_label = {files.name};
for i = 1:67
    lenet_label(i)=sscanf(image_train_label{i},'%d');
end

for i = 1:67
    
    %lenet_label2(i) = hwLabel(lenet_label(i));
    labels2(lenet_label(i)) = labels(i);
end


