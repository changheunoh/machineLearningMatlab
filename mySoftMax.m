function resultArray = mySoftMax(array, type)

total_array_num = length(array);

accum = 0;
for ii=1:total_array_num
    accum = accum + exp( array(ii) );
end

% resultArray = zeros( size(array,1), size(array,2) );
resultArray = zeros( size(array,1), 1 );
for ii=1:total_array_num
   resultArray(ii) = exp( array(ii) )/accum;
end
    
end