function [ all_proba ] = softmax( X, W, b )

    [~, n_samples] = size(X);
    [~, n_classes] = size(W);

    log_numer = W' * X + repmat(b, 1, n_samples);
    max_log_numer = max(log_numer, [], 1);
    log_numer = log_numer - repmat(max_log_numer, n_classes, 1);
    numerator = exp(log_numer);
    denominator = sum(numerator, 1);
    
    all_proba = numerator * diag(denominator.^(-1));
end

