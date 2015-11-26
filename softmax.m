function [ output ] = softmax( input )
    exp_input = exp(input);
    output = exp_input / sum(exp_input);
end

