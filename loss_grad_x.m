function [ g ] = loss_grad_x( X, C, W, b )
    [~, n_samples] = size(X);
    
    log_numer = X' * W + repmat(b', n_samples, 1);
    max_log_numer = max(log_numer, [], 2);
    log_numer = bsxfun(@minus, log_numer, max_log_numer);  % log-sum-exp trick
    numerator = exp(log_numer);
    denominator = sum(numerator, 2);
    
    all_proba = diag(denominator.^(-1)) * numerator; % row = p(label) per x
    
    g = W * (all_proba' - C);
end
