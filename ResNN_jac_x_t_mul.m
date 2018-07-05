function [ res ] = ResNN_jac_x_t_mul( X, W1, W2, b, v )
    % X - batch with size=(dim, batch_size)
    % W1, W2, b - layer's weights
    % V - batch of vectors of size=(dim, batch_size)
    %
    % returns res with size=(dim, batch_size)
    %   each col in res is the result of the jacobian at X_i transpose
    %   times V_i (X_i and V_i being the i-th columns of X and V)
    
    
    [~, m] = size(X);

    inner = W1 * X + repmat(b, 1, m);
    sig_inner = sigmoid(inner);
    der_sig_inner = sig_inner .* (1-sig_inner);
    
    right_side = W1' * (der_sig_inner .* (W2' * v));
    
    res = v + right_side;
end

