function [ y ] = der_sigmoid( x, w )
    sig_x = sigmoid(x, w);
    y = (1 - sig_x) * sig_x;
end

