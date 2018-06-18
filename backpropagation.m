function [ theta_grad ] = backpropagation( X, C, Theta, Xis )

    

end

function [ res ] = mul_jac_b( X, W1, W2, b, v )
    
    [~, m] = size(X);

    inner = W1 * X + repmat(b, m);
    outer = W2' * v;
    
    sig_inner = arrayfun(@sigmoid,inner);
    der_sig_inner = sig_inner .* (1-sig_inner);
    
    m_outer = repmat(outer, 1, m);
    prod = der_sig_inner .* m_outer;  % n times m
    res = (1 / m) * sum(prod, 2);  % avg between cols
end

function [ res ] = mul_jac_W1( X, W1, W2, b, v )
    
    [~, n_samples] = size(X);
    
end

function [ res ] = mul_jac_W2( X, W1, b, v )
    
    [~, m] = size(X);
    [n, ~] = size(W1);

    inner = W1 * X + repmat(b, m);
    sig_inner = sigmoid(inner);
    
    flat_sig_inner = sig_inner(:);  % n*m times 1
    
    prod = v * flat_sig_inner';  % n times n*m
    reshaped_prod = reshape(prod, [n, n, m]);  % n times n times m
    
    res = (1 / m) * sum(reshaped_prod, 3);  % avg over 3rd axis
end
