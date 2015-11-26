data = load('MNIST.mat');

num_h_layers = 3;
num_h_neurons(1,1) = 100;
num_h_neurons(2,1) = 20;
num_h_neurons(3,1) = 10;

type= 'ReLU';
% type= 'sigmoid';
size_input = length( data.test_data(1,:) );

% random initialize
W = cell(num_h_layers,1);
for ii=1:num_h_layers

    if (ii==1)
        size_w = size_input+1;
    else
        size_w = num_h_neurons(ii-1,1)+1;
    end

    W{ii} = zeros(size_w,num_h_neurons(ii,1));

    % sqrt(2/model(i))*randn(model(i)+1,model(i+1))
    W{ii} = sqrt(2/(size_w-1))*randn( size_w, num_h_neurons(ii,1) );

end

total_num = size(data.test_data,1);
index = randperm(total_num);
index_5000 = index(1:5000);

myData.train_data = data.train_data(index_5000,:);
myData.train_label = data.train_label(index_5000,:);
myData.test_data = data.test_data;
myData.test_label = data.test_label;



%%

W = myTraining_mini(myData, W, type);


%%

% accuracyPercentTrain = 100* myTestAccuracy(data.train_data, data.train_label, W, type)
% accuracyPercentTest = 100* myTestAccuracy(data.test_data, data.test_label, W, type)
%% save W.mat W
%% report training log history(losses, training errors, test errors for each interation)

load('TestError.mat');
load('TrainingError.mat');

figure, plot(1:10, TestError), ylim([0 100]),
hold on, plot(1:10, TrainingError)

load('MiniBatchLoss.mat')
figure, plot(1:length(MiniBatchLoss), MiniBatchLoss)