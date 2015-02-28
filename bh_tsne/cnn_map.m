%---------This File is to map the features to 2-D dimension----------%
load('../imagenet_feature.mat');
load('../train_label.mat');
load('../test_feature.mat');
load('../test_label.mat');
features = [feature_new(1:14360,:);test_feature];
labels = [train_label; test_label];
no_dims = 2;
initial_dims = 30;
perplexity = 30;
fprintf('initialization is ok now');
[mappedX] = fast_tsne(features, no_dims, initial_dims, perplexity);
gscatter(mappedX(:,1),mappedX(:,2),labels);