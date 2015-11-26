function [A,H] = myFeedForward(data, W, type)


% input 1x28*28
% 'tanh' 'sigmoid' 'ReLU'
% type= 'ReLU';

num_h_layers = length(W);
H = cell(num_h_layers,1);
A = cell(num_h_layers,1);

totalsize = size(data,1)*size(data,2);
data = reshape(data, 1,totalsize);

for i = 1:num_h_layers-1
    if(i==1)
        input = [1; data'];
    else
        input = [1; A{i-1,1}];
    end
        
    
    H{i,1} = W{i,1}'*input;
    A{i,1} = mySigmoid(H{i,1}, 1, type);
end

H{end,1} = W{end,1}'*[1; A{end-1,1}];
A{end,1} = mySoftMax(H{end,1}, type);


