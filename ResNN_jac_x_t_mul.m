function [ res ] = ResNN_jac_x_t_mul( X, W1, W2, b, v )
    
    [~, m] = size(X);

    inner = W1 * X + repmat(b, 1, m);
    sig_inner = sigmoid(inner);
    der_sig_inner = sig_inner .* (1-sig_inner);
    
    right_side = W1' * (der_sig_inner .* (W2' * v));
    
    res = v + right_side;
end

