function [ X_next ] = ResNN( X, W1, W2, b )
    [~, n_samples] = size(X);

    inner = W1 * X + repmat(b, 1, n_samples);
    X_next = X + W2 * sigmoid(inner);
end
