function [ g ] = loss_grad_theta( X, C, W, b )
    
    [~, n_samples] = size(X);

    all_proba = softmax(X, W, b);
    
    g_W = X * (all_proba - C');
    g_b = ones(1, n_samples) * (all_proba - C');
    
    g = vertcat(g_b(:), g_W(:));
end

