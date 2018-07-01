function [ g ] = loss_grad_theta( X, C, W, b )
    [~, n_samples] = size(X);

    log_numer = X' * W + repmat(b', n_samples, 1);
    max_log_numer = max(log_numer, [], 2);
    log_numer = bsxfun(@minus, log_numer, max_log_numer);  % log-sum-exp trick
    numerator = exp(log_numer);
    denominator = sum(numerator, 2);
    
    all_proba = diag(denominator.^(-1)) * numerator;
    
    g_W = X * (all_proba - C');
    g_b = ones(1, n_samples) * (all_proba - C');
    
    g = vertcat(g_b(:), g_W(:));
end

