function [ X_next ] = ResNN( X, W1, W2, b )
    % X - batch with size=(dim, batch_size)
    % W1, W2, b - layer's weights
    
    
    [~, n_samples] = size(X);

    inner = W1 * X + repmat(b, 1, n_samples);
    X_next = X + W2 * sigmoid(inner);
end
