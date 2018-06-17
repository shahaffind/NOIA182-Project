function [ loss_val ] = loss( X, C, W )
    
    log_numer = X' * W;
    max_log_numer = max(log_numer, [], 2);
    log_numer = bsxfun(@minus, log_numer, max_log_numer);
    numerator = exp(log_numer);
    denominator = sum(numerator, 2);
    
    all_proba = diag(denominator.^(-1)) * numerator;
    
    relevant_proba = C' .* log(all_proba + eps);
    
    loss_val = sum(relevant_proba(:));
    
end

