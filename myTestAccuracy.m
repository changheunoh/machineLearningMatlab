function accuracyPercent = myTestAccuracy(test_data, test_label, W, type)

% test_index = 387;
% [A,H] = myFeedForward(data.test_data(test_index,:),W);
% disp(A{3,1}');
% disp(data.test_label(test_index, : ));

size_test = size(test_data, 1);
cnt_true =0;

for index = 1:size_test
    
    [A,~] = myFeedForward(test_data(index,:),W, type);
    
    [~,I_predicted] = max( A{end,1}' );
    [~,I_label] = max( test_label(index,:) );
    
    if(I_predicted==I_label)
       cnt_true = cnt_true + 1; 
    end
    
end


accuracyPercent = cnt_true  /  size_test ;


end