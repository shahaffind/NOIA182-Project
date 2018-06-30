function [ X_next ] = ResNN( X, W1, W2, b )
    X_next = X + W2 * sigmoid(X'*W1 + b);
end
