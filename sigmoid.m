function [ y ] = sigmoid( x, w )
    y = 1 / (1 + exp(-x'*w));
end

