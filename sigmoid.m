function [ y ] = sigmoid( x, w, b )
    y = 1 / (1 + exp(-x'*w + b));
end

