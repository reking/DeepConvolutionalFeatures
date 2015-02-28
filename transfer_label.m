load('caffetestlabel.txt');
image_test_label=load('image_test_label.mat');
image_train_label1=image_test_label.name;
image_train_label2=zeros(5266,1);
test_label=zeros(5226,1);
for i = 1:5226
    image_train_label2(i)=sscanf(image_train_label1{i},'%d');
end

for i = 1:5226
    
    test_label(i) = caffetestlabel(image_train_label2(i)+1,2);
end
save('test_label.mat','test_label');