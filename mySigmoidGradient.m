function a_prime = mySigmoidGradient(hh, alpha, type)

if strcmp(type, 'sigmoid')
    exp_term = exp(-alpha*hh);
    sigmoid = 1 ./ (1+exp_term);
    a_prime = sigmoid.*(1-sigmoid); % when alpha =1;
    
elseif strcmp(type, 'tanh')
    exp_term = exp(-alpha*hh);
%     tanh = (1-exp_term) ./ (1+exp_term);
    a_prime = (2*alpha*exp_term.*exp_term)./(( 1+exp_term ).^2);
    
elseif strcmp(type, 'ReLU')        
    a_prime = hh;
    a_prime(hh > 0) = 1;
    a_prime(hh <= 0) = 0;
end

