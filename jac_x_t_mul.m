function [ res ] = jac_x_t_mul( X, W1, W2, b, v )
    
    [~, m] = size(X);

    inner = W1 * X + repmat(b, 1, m);
    sig_inner = arrayfun(@sigmoid,inner);
    der_sig_inner = sig_inner .* (1-sig_inner);
    
    right_size = W1' * (der_sig_inner .* (W2' * v));
    
    res = v + right_size;
    res = (1/m) * sum(res, 2);  % avg over all X
    
end

