function a = mySigmoid(hh, alpha, type)

if strcmp(type, 'sigmoid')
    exp_term = exp(-alpha*hh);
    a = 1 ./ (1+exp_term);
elseif strcmp(type, 'tanh')
    exp_term = exp(-alpha*hh);
    a = (1-exp_term) ./ (1+exp_term);
    
elseif strcmp(type, 'ReLU')
    a = max(0, hh);    
end



