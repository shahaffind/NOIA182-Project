function [ res ] = ResNN_jac_x_mul( x, W1, W2, b, v )
    
    [n, ~] = size(W1);

    inner = W1 * x + b;
    sig_inner = arrayfun(@sigmoid,inner);
    der_sig_inner = sig_inner .* (1-sig_inner);
    
    jac = eye(n) + W2 * diag(der_sig_inner) * W1;
    
    res = jac * v;
    
end
