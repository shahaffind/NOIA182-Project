function [ all_proba ] = softmax( X, W, b )

    [~, n_samples] = size(X);
    [~, n_classes] = size(W);

    log_numer = X' * W + repmat(b', n_samples, 1);
    max_log_numer = max(log_numer, [], 2);
    log_numer = log_numer - repmat(max_log_numer, 1, n_classes);
    numerator = exp(log_numer);
    denominator = sum(numerator, 2);
    
    all_proba = diag(denominator.^(-1)) * numerator;

end

