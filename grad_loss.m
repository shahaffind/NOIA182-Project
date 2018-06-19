function [ g ] = grad_loss( X, C, W )
    log_numer = X' * W;
    max_log_numer = max(log_numer, [], 2);
    log_numer = bsxfun(@minus, log_numer, max_log_numer);
    numerator = exp(log_numer);
    denominator = sum(numerator, 2);
    
    all_proba = diag(denominator.^(-1)) * numerator;
    
    g = X * (all_proba - C');    
    
end

