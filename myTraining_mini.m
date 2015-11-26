function W = myTraining_mini(myData, W, type)
train_data = myData.train_data;
train_label = myData.train_label;
test_data = myData.test_data;
test_label = myData.test_label;



max_iter = 10;   mini = 100;
MSE = [];       total_num = size(train_data,1);     num_h_layers = length(W);

lr=1.5E-3;
MSE_temp=0;
MiniBatchLoss=[];


for epoch=1:max_iter
%     messege = ['Please wait...      ', 'epoch : ', num2str(epoch)];     h = waitbar(0,messege);
    RealTimeMSE = [];
    lr = 0.9*lr;

    for ii = 1:mini:total_num

        delta_W= cell(num_h_layers, 1);
        loss_mini_batch = 0;
        for jj=1:num_h_layers
           delta_W{jj,1} = zeros(size(W{jj,1}));
        end
        lr = 1*lr;

        for mm=0:mini
            if(mm==(mini))
                messegeW = ['[',num2str(norm([norm(W{1}), norm(W{2})])), ' , ',num2str(lr*norm([norm(delta_W{1}), norm(delta_W{2})])),']' ];
                messegeLoss = [9, 'loss_mini_batch : ', num2str(loss_mini_batch/mini)];
                %%% in Matlab, disp not work with '\t', so, ascii code of
                %%% tab(\t) = 9;
                MiniBatchLoss(end+1)=loss_mini_batch/mini;
                
                disp(strcat(messegeW, messegeLoss));
                
                for kk=1:num_h_layers
                    W{kk,1} = W{kk,1} + lr*delta_W{kk,1};
                    delta_W{kk,1} = 0*delta_W{kk,1};
                end
                continue;
            end


            index = ii+mm;
            if(index>total_num)
                continue;
            end

            temp_input = train_data(index,:);
            [A, H] = myFeedForward(temp_input, W, type);
%             error = - (train_label(index,:)' - A{end,:}); % sign??
            error = (train_label(index,:)' - A{end,:}); % sign??

            for jj=num_h_layers:-1:1
                a_prime{jj,1} = mySigmoidGradient(H{jj,1}, 1, type);

                if (jj==num_h_layers)
                    delta{jj,1} = error.*a_prime{end,1};
                else
                    delta{jj,1} = (W{jj+1,1}(2:end,:)*delta{jj+1,1}).*a_prime{jj,1};
                end
            end

            for jj=1:num_h_layers
                delta_W{jj,1}(1,:) = delta_W{jj,1}(1,:) + delta{jj,1}';
                if(jj~=1)
                    delta_W{jj,1}(2:end,:) = delta_W{jj,1}(2:end,:) + (delta{jj,1}*A{jj-1,1}')';
                else
                    delta_W{jj,1}(2:end,:) = delta_W{jj,1}(2:end,:) + (delta{jj,1}*temp_input)';
                end
            end

            % MSE_temp = MSE_temp + sum(error.^2);
            loss_mini_batch = sqrt( loss_mini_batch^2 + sum(error.^2) );

%             waitbar( index / total_num);
            RealTimeMSE(ceil(index/mini)) = sum(error.^2);

        end


    end
    % figure(epoch), plot(1:length(RealTimeMSE), RealTimeMSE);

    % MSE(end+1,1) = MSE_temp/total_num;
    % MSE_temp = 0;

    % messege = ['epoch ', num2str(epoch), '    MSE ', num2str(MSE(end))];
    % disp(messege);
%     close(h);

    TrainingError(epoch) = 100* myTestAccuracy(myData.train_data, myData.train_label, W, type);
    TestError(epoch) = 100* myTestAccuracy(myData.test_data, myData.test_label, W, type);
    disp(['epoch :',num2str(epoch),9,'Trn : ', num2str(TrainingError(epoch)),9, 'Test : ', num2str(TestError(epoch))]);

end

save TrainingError.mat TrainingError;
save TestError.mat TestError;
save MiniBatchLoss.mat MiniBatchLoss;

% TrainingError = 100* myTestAccuracy(data.train_data, data.train_label, W, type)
% TestError = 100* myTestAccuracy(data.test_data, data.test_label, W, type)
end
