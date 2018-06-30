function [ res ] = ResNN_jac_theta_t_mul( X, W1, W2, b, v )
    
    [n, m] = size(X);
    
    JbtV = mul_jac_b_t(X, W1, W2, b, v);
    JbtV_avg = (1/m) * sum(JbtV, 2);
    
    Jw1tV = mul_jac_W1_t(X, JbtV);  % n times n*m
    Jw1tV = reshape(Jw1tV, [n, n, m]);  % n times n times m
    Jw1tV_avg = (1 / m) * sum(Jw1tV, 3);  % avg over 3rd axis, now n times n
    Jw1tV_avg = Jw1tV_avg(:);
    
    Jw2tV = mul_jac_W2_t(X, W1, b, v);  % n times n*m
    Jw2tV = reshape(Jw2tV, [n, n, m]);  % n times n times m
    Jw2tV_avg = (1 / m) * sum(Jw2tV, 3);  % avg over 3rd axis, now n times n
    Jw2tV_avg = Jw2tV_avg(:);
    
    res = vertcat(JbtV_avg, Jw1tV_avg, Jw2tV_avg);
end

function [ res ] = mul_jac_b_t( X, W1, W2, b, v )
    [~, m] = size(X);

    inner = W1 * X + repmat(b, 1, m);
    outer = W2' * v;
    
    sig_inner = arrayfun(@sigmoid,inner);
    der_sig_inner = sig_inner .* (1-sig_inner);
    
    m_outer = repmat(outer, 1, m);
    res = der_sig_inner .* m_outer;  % n times m
end

function [ res ] = mul_jac_W1_t( X, JbtV )
    [n, m] = size(X);
    
    V_rep = repmat(JbtV, n, 1);
    
    X_temp = X';
    X_temp = repmat(X_temp, n, 1);
    X_temp = reshape(X_temp, [m, n^2]);
    X_temp = X_temp';
    
    res = X_temp .* V_rep;  % n times n*m
end

function [ res ] = mul_jac_W2_t( X, W1, b, v )
    [~, m] = size(X);
    
    inner = W1 * X + repmat(b, 1, m);
    sig_inner = arrayfun(@sigmoid,inner);
    
    flat_sig_inner = sig_inner(:);  % n*m times 1
    
    res = v * flat_sig_inner';  % n times n*m
end