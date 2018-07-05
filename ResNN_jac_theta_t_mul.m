function [ res ] = ResNN_jac_theta_t_mul( X, W1, W2, b, V )
    % X - batch with size=(dim, batch_size)
    % W1, W2, b - layer's weights
    % V - batch of vectors of size=(dim, batch_size)
    %
    % returns res with size=(number of weights, 1)
    %   res is the stochastic result of the jacobian (w.r.t. weights) at X_i
    %   transpose times V_i (X_i and V_i being the i-th columns of X and V)


    [n, m] = size(X);
    
    inner = W1 * X + repmat(b, 1, m);
    sig_inner = sigmoid(inner);
    der_sig_inner = sig_inner .* (1 - sig_inner);
    
    %%%% derivative wrt b transpose times v
    JbtV = der_sig_inner .* (W2' * V);  % does the same effect of "diag" operation
    JbtV_avg = (1 / m) * sum(JbtV, 2);  % stochastic result
    
    %%%% derivative wrt W1 transpose times v
    % the following does the same effect of (X' kron I) multiplication
    X_rep = repmat(X', n, 1);
    X_rep = reshape(X_rep, [m, n^2]);
    JbtV_rep = repmat(JbtV, n, 1); 
    Jw1tV = X_rep' .* JbtV_rep; % n^2 times m
    Jw1tV_avg = (1 / m) * sum(Jw1tV, 2);  % stochastic result
    
    %%%% derivative wrt W2 transpose times v
    % (comes as sum of all derivatives per x,v pair)
    Jw2tV_sum = V * sig_inner';
    Jw2tV_avg = Jw2tV_sum(:) / m;  % stochastic result
    
    res = vertcat(JbtV_avg, Jw1tV_avg, Jw2tV_avg);
end