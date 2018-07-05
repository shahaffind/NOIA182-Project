function [ loss_val ] = loss( X, C, W, b )
    % X - batch with size=(dim, batch_size)
    % C - correct result column for each X_i, size=(n_labels, batch_size)
    % W, b - loss layer weights
    %
    % returns the loss value of the batch
    
    
    all_proba = softmax(X, W, b);
    
    relevant_proba = C' .* log(all_proba + eps);
    loss_val = - sum(relevant_proba(:));
end

