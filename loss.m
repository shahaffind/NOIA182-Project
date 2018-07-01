function [ loss_val ] = loss( X, C, W, b )
    
    all_proba = softmax(X, W, b);
    
    relevant_proba = C' .* log(all_proba + eps);
    
    loss_val = - sum(relevant_proba(:));
    
end

